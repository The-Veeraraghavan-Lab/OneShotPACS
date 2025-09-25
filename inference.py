#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import argparse
import numpy as np
import torch

import SimpleITK as sitk

from scipy.ndimage import label as label_connected_components
from scipy.ndimage import generate_binary_structure


from monai import data, transforms
# from monai.inferers import sliding_window_inference
from monai.data import  decollate_batch ,load_decathlon_datalist
from monai.transforms import (
    EnsureTyped,
    Invertd,
)
from monai.handlers.utils import from_engine

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import model.voxelmorph as vxm

from utils_data import get_val_transforms_base, copy_info
from utils_metrics import Eval

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--data_dir', default='dataset', type=str, help='root directory cointaining the dataset and JSON lists.')
parser.add_argument('--json_list', default='example.json', type=str, help='dataset json file listing cases.')
parser.add_argument("--sv_dir", type=str, default="sv_dir", help='Root output directory for results, metrics, and NIfTI saves.')
parser.add_argument("--val_dir", type=str, default="inference_190", help='Subdirectory under --sv_dir for current run’s outputs.')

# models and weights
parser.add_argument('--reg_model', type=str, default="VoxelMorph", choices=["VoxelMorph"],
                help='Registration model (Only VoxelMorph is supported).')
parser.add_argument('--reg_weights', type=str, default='saved_weights/0190_reg.pt'
                help='Path (relative to --sv_dir) to registration model weights.')
parser.add_argument('--seg_model', type=str, default="CLSTM", choices=["CLSTM", "none"],
                help='Segmentation head (Use 'none' to disable segmentation).')
parser.add_argument('--seg_weights', default='saved_weights/0190_seg.pt',
                help='Path (relative to --sv_dir) to segmentation model weights.')

# registration model hyperparameters
parser.add_argument('--enc_nf', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32).')
parser.add_argument('--dec_nf', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16).')
parser.add_argument('--int_steps', type=int, default=7, help='number of integration steps (default: 7).')
parser.add_argument('--int_downsize', type=int, default=2, help='flow downsample factor for integration (default: 2).')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function.')
parser.add_argument('--flownum', type=int, default=8, help='flow number (default: 8)'.)

# preprocessing parameters
parser.add_argument('--a_min', default=-500, type=float, help='a_min in ScaleIntensityRanged.')
parser.add_argument('--a_max', default=500, type=float, help='a_max in ScaleIntensityRanged.')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged.')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged.')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction.')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction.')
parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction.')
parser.add_argument('--roi_x', default=256, type=int, help='roi size in x direction.')
parser.add_argument('--roi_y', default=256, type=int, help='roi size in y direction.')
parser.add_argument('--roi_z', default=64, type=int, help='roi size in z direction.')

# others
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--workers', default=1, type=int, help='number of workers')
parser.add_argument('--cudnn-nondet',  action='store_true', help='disable cudnn determinism - might slow down training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--no_img_sv', action='store_true', help='Code will not save output images')
parser.add_argument('--affine', action='store_true', help='Add Affine rotations to the input images')
parser.add_argument('--rotate', default='0.0', help='Rotation in degrees to apply to affine transform')
parser.add_argument('--trans', default='0', help='Translation in voxel size to apply to affine transform')

args = parser.parse_args()
print (args)

# device handling
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet
torch.backends.cudnn.benchmark     = args.cudnn_nondet

# Load the dataset
datalist_json = os.path.join(args.data_dir, args.json_list)

test_files = load_decathlon_datalist(datalist_json,
                                     True,
                                     "validation",
                                     base_dir = args.data_dir
                                     )

val_transform = get_val_transforms_base(args)
val_org_ds = data.Dataset(data=test_files, transform=val_transform)
val_org_loader = data.DataLoader(val_org_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=(device.type == 'cuda'))

print('val data size is ',len(val_org_loader))

post_transforms = transforms.Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=val_transform,
        orig_keys="cbct",
        meta_keys="pred_meta_dict",
        orig_meta_keys="cbct_meta_dict",
        meta_key_postfix="meta_dict",
        #nearest_interp=True,
        to_tensor=True,
    ),
    #AsDiscreted(keys="pred", argmax=True),
    
])

post_transforms_dvf = transforms.Compose([
    EnsureTyped(keys="dvf"),
    Invertd(
        keys="dvf",
        transform=val_transform,
        orig_keys="cbct",
        meta_keys="dvf_meta_dict",
        orig_meta_keys="cbct_meta_dict",
        meta_key_postfix="meta_dict",
        #nearest_interp=True,
        to_tensor=True,
    ),
    #AsDiscreted(keys="pred", argmax=True),
])

post_transforms_contour = transforms.Compose([
    EnsureTyped(keys="pred_m"),
    Invertd(
        keys="pred_m",
        transform=val_transform,
        orig_keys="cbct_msk",
        meta_keys="pred_m_meta_dict",
        orig_meta_keys="cbct_msk_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    ),
    #AsDiscreted(keys="pred", argmax=True),
    
])

###############################################################################

# Define the model and load the corresponding weights
# unet architecture
args.enc_nf = args.enc_nf if args.enc_nf else [16, 32, 32, 32]
args.dec_nf = args.dec_nf if args.dec_nf else [32, 32, 32, 32, 32, 16, 16]

# Define registration model -------------

if args.reg_model == 'VoxelMorph':
    reg_model = vxm.networks.VxmDense_3D_LSTM_Step_Reg_All_Encoder_LSTM(  
        inshape=(args.roi_x, args.roi_y, args.roi_z),
        nb_unet_features=[args.enc_nf, args.dec_nf],
        bidir=args.bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        range_flow=10,
    )
else:
    raise NotImplementedError("Only VoxelMorph is supported at this time.")

reg_model_path = os.path.join(args.sv_dir, args.reg_weights)
if not os.path.isfile(reg_model_path):
    raise FileNotFoundError(f"Registration weights not found at {reg_model_path}")
reg_model.load_state_dict(torch.load(reg_model_path, map_location="cpu"))
reg_model.to(device)
reg_model.eval()

# Done defining registration model -------------

# Define segmentation model -------------
use_seg = (args.seg_model is not None) and (str(args.seg_model).lower() != 'none')

if use_seg:
    if args.seg_model == 'CLSTM':
        seg_model = vxm.networks.UNet3D_Seg_LSTM(in_channels=3,out_channels=2+1,final_sigmoid=False)
    else:
        raise NotImplementedError("Only CLSTM is supported at this time.")

    seg_model_path = os.path.join(args.sv_dir, args.seg_weights)
    if not os.path.isfile(seg_model_path):
        raise FileNotFoundError(f"Segmentation weights not found at {seg_model_path}")

    seg_ckpt = torch.load(seg_model_path, map_location="cpu")
    seg_state = seg_ckpt.get('model_state_dict', seg_ckpt)
    seg_model.load_state_dict(seg_state)
    seg_model.to(device)
    seg_model.eval()
else:
    print("Not using segmentation")
        
# Done defining segmentation model -------------


###############################################################################

val_folder = os.path.join(args.sv_dir, args.val_dir) 

with torch.no_grad(): # no grade calculation  
    class_list = ['Eso', 'GTV']
    eval = Eval(val_folder, class_list, mode='test')

    for i_iter_val, batch_data in enumerate(val_org_loader):    
                            
        plan_ct_img, planct_val_msk = batch_data['pct'].float().to(device), batch_data['pct_msk'].float().to(device)
        cbct_val_img, cbct_val_msk = batch_data['cbct'].float().to(device), batch_data['cbct_msk'].float().to(device)

        flow_in=torch.zeros(args.batch_size, 3, cbct_val_img.size()[2], cbct_val_img.size()[3], cbct_val_img.size()[4]).to(device)

        p_name = batch_data['cbct_meta_dict']['filename_or_obj'][0].split('/')[-2]
        print(p_name)
        
        cur_folder=os.path.join(val_folder,p_name) + '/'
        sv_folder = os.path.join(val_folder,p_name)+'/'

        if not args.no_img_sv:
            if not os.path.exists(sv_folder):
                os.makedirs(sv_folder, exist_ok=True)
        else:
            sv_folder = 'None'
        
        # recurrent states
        h = c = None
        state_seg = None

        # core iterative registration/seg
        for seg_iter_val in range(args.flownum):
            if seg_iter_val == 0:
                y_pred_val, y_m_pred_val, dvf_flow, h, c = reg_model.forward_seg_training_all_enc_lstm_accu_dvf(
                    plan_ct_img, cbct_val_img, planct_val_msk, h, c, flow_in, plan_ct_img, planct_val_msk
                )
            else:
                y_pred_val, y_m_pred_val, dvf_flow, h, c = reg_model.forward_seg_training_all_enc_lstm_accu_dvf(
                    y_pred_val, cbct_val_img, y_m_pred_val, h, c, dvf_flow, plan_ct_img, planct_val_msk
                )
 
            if args.seg_model is not None:                    
                # seg input: [cbct, deformed_moving_img, deformed_moving_mask]
                seg_in_val = torch.cat((cbct_val_img, y_pred_val, y_m_pred_val), dim=1)
                seg_logits, h_seg, c_seg = seg_model(seg_in_val, state_seg)
                state_seg = [h_seg, c_seg]
                seg_result = torch.argmax(seg_logits, dim=1)

        # Loading reference image
        pct_path = batch_data['pct_meta_dict']['filename_or_obj'][0]
        pct_ref = sitk.ReadImage(pct_path) 

        # Deformed Image and Mask Transform (Removing all transforms)
        batch_data["pred"] = y_pred_val
        batch_data["dvf"] = dvf_flow
        batch_data["pred_m"] = y_m_pred_val
    
        batch_data_img_def = [post_transforms(i) for i in decollate_batch(batch_data)]
        batch_data_img_dvf = [post_transforms_dvf(i) for i in decollate_batch(batch_data)]
        batch_data_msk_def = [post_transforms_contour(i) for i in decollate_batch(batch_data)]

        y_pred_val = from_engine(["pred"])(batch_data_img_def)[0]   
        dvf_flow = from_engine(["dvf"])(batch_data_img_dvf)[0]
        y_m_pred_val = from_engine(["pred_m"])(batch_data_msk_def)[0]   

        # CBCT Image Mask Transform
        batch_data["pred"] = cbct_val_img
        batch_data["pred_m"] = cbct_val_msk

        batch_data_img_def = [post_transforms(i) for i in decollate_batch(batch_data)]
        batch_data_msk_def = [post_transforms_contour(i) for i in decollate_batch(batch_data)] 

        cbct_val_img = from_engine(["pred"])(batch_data_img_def)[0]   
        cbct_val_msk = from_engine(["pred_m"])(batch_data_msk_def)[0]   
        
        # planCT Image Mask Transform
        batch_data["pred"] = plan_ct_img
        batch_data["pred_m"] = planct_val_msk

        batch_data_img_def = [post_transforms(i) for i in decollate_batch(batch_data)]
        batch_data_msk_def = [post_transforms_contour(i) for i in decollate_batch(batch_data)] 

        plan_ct_img = from_engine(["pred"])(batch_data_img_def)[0]   
        planct_val_msk = from_engine(["pred_m"])(batch_data_msk_def)[0] 
        
        info = {'Image_Name':p_name}
        spacing = pct_ref.GetSpacing()

        if args.seg_model is not None:
            batch_data["pred_m"] = torch.unsqueeze(seg_result, 1)
    
            batch_data_msk_def = [post_transforms_contour(i) for i in decollate_batch(batch_data)] 
            seg_result = from_engine(["pred_m"])(batch_data_msk_def)[0] 

            ## Removing excess labelling in segemntation label
            num = [750, 4000] #$ Hard-coded for now to assume values for esophagus and GTV
            
            temp_array = torch.zeros_like(torch.squeeze(seg_result))
            
            for i in range(1,3):
                temp = np.squeeze(np.copy(seg_result))
                temp[temp!=i] = 0
                # print(np.unique(temp,return_counts=True))
                pred_mask_lesion, num_predicted = label_connected_components(temp,structure=generate_binary_structure(3, 3),output=np.int16)
                component_counts = np.bincount(pred_mask_lesion.flatten())
                # print("Seg Count")
                # print(component_counts)
                valid_components = np.where(component_counts > num[i-1])[0]
                
                pred_volume2 = np.where(np.isin(pred_mask_lesion, valid_components), pred_mask_lesion, 0)
    
                pred_mask_lesion, num_predicted = label_connected_components(pred_volume2, structure=generate_binary_structure(3, 3),output=np.int16)
                # if len(np.unique(pred_mask_lesion)) > :
                temp_array[torch.from_numpy(pred_mask_lesion)==1] = i
            seg_result = temp_array
            
            eval.calculate_results(info, spacing, cbct_val_msk, planct_val_msk, y_m_pred_val, seg_result, dvf_flow)
        else:
            eval.calculate_results(info, spacing, cbct_val_msk, planct_val_msk, y_m_pred_val, None, dvf_flow)
            
        if not args.no_img_sv:
            
            plan_img_data=plan_ct_img.float().cpu().numpy()
            plan_msk_data=planct_val_msk.float().cpu().numpy()
            cbct_img_data=cbct_val_img.float().cpu().numpy()
            cbct_msk_data=cbct_val_msk.float().cpu().numpy()
            plan_def_data=y_pred_val.float().cpu().numpy()
            reg_result=y_m_pred_val.float().cpu().numpy()

            #$ If needed, uncomment these lines to also allow for moving the following 
            #$ variables to CPU, then add lines to save them to hard drive
            # dvf_flow = dvf_flow.float().cpu().numpy()
            
            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(plan_img_data),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'planCT.nii'
            sitk.WriteImage(save_img, save_path)

            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(plan_msk_data),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'planCT_msk.nii'
            sitk.WriteImage(save_img, save_path)

            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(cbct_img_data),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'CBCT.nii'
            sitk.WriteImage(save_img, save_path)

            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(cbct_msk_data),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'CBCT_msk.nii'
            sitk.WriteImage(save_img, save_path)

            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(plan_def_data),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'planCT_def.nii'
            sitk.WriteImage(save_img, save_path)

            save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(reg_result),(2,1,0)))
            save_img = copy_info(pct_ref, save_img)
            save_path = sv_folder+'planCT_def_msk.nii'
            sitk.WriteImage(save_img, save_path)

            if args.seg_model is not None:
                seg_result=seg_result.float().cpu().numpy()

                save_img = sitk.GetImageFromArray(np.transpose(np.squeeze(seg_result),(2,1,0)))
                save_img = copy_info(pct_ref, save_img)
                save_path = sv_folder+'cbct_seg_msk.nii'
                sitk.WriteImage(save_img, save_path)