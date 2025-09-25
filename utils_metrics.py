import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd
import numpy as np
import os
from medpy.metric import binary

from metrics.HD import compute_surface_distances, compute_robust_hausdorff

# prepare deformation loss
def calculate_dice(ground_truth, pred, num_classes):
    """
    Calculates the Dice coefficient between ground_truth and pred arrays.
    
    Args:
        ground_truth (np.ndarray): Ground truth array of shape (512, 512, 50).
        pred (np.ndarray): Prediction array of shape (512, 512, 50).
    
    Returns:
        class_dice (np.ndarray): Array containing class-wise Dice coefficients.
        mean_dice (float): Mean average Dice coefficient.
    """

    # Flatten the arrays along the batch and channel dimensions
    ground_truth = np.reshape(ground_truth, (-1,))
    pred = np.reshape(pred, (-1,))
    # Calculate the Dice coefficient for each class
    # num_classes = 3#np.max(ground_truth) + 1
    metric_dice = []
    # print(np.unique(ground_truth))
    # print(np.unique(pred))
    # print("CLASSES:", num_classes)
    for class_label in range(1,num_classes):
        # Create binary masks for the current class
        gt_mask = (ground_truth == class_label)
        pred_mask = (pred == class_label)
        if np.any(gt_mask): #and np.any(pred_mask):
            dice_x = binary.dc(pred_mask, gt_mask)
            metric_dice.append(dice_x)
        else:
            metric_dice.append(np.nan)
    
    # Calculate mean average Dice coefficient
    metric_dice.append(np.nanmean(metric_dice))
    
    return metric_dice

def calculate_hd_95(ground_truth, pred, spacing, num_classes):
    # num_classes = 3#np.max(ground_truth) + 1
    ground_truth = np.squeeze(ground_truth)
    pred = np.squeeze(pred)
    metric_hd95 = []
    for class_label in range(1,num_classes):
        gt_mask = (ground_truth == class_label)
        pred_mask = (pred == class_label)
        surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing)
        hd_95 = compute_robust_hausdorff(surface_distances, 95.0)
        metric_hd95.append(hd_95)
    metric_hd95.append(np.nanmean(metric_hd95))
    return metric_hd95


def calculate_displacement(dvf_flow, pred, num_classes):
    # num_classes = 3
    # print(pred.shape)
    # print(dvf_flow.shape)
    metrics_disp = []
    pred = np.squeeze(pred)
    dvf_flow = np.squeeze(dvf_flow)
    for class_label in range(1,num_classes):
        mask = np.where(pred==class_label)
        a = dvf_flow[0][mask]
        b = dvf_flow[1][mask]
        c = dvf_flow[2][mask]
        temp = np.square(a)+np.square(b)+np.square(c)
        metrics_disp.append(np.nanmean(np.sqrt(temp)))
    metrics_disp.append(np.nanmean(metrics_disp))
    return metrics_disp

class Eval:
    def __init__(self, sv_dir, class_list, calc_reg=True, calc_seg=True, calc_disp=True, mode='train'):
        self.df = pd.DataFrame()
        self.avg_df = pd.DataFrame()
        self.epoch = 0
        
        self.sv_dir = sv_dir
        class_list.append('Avg')
        self.class_list = class_list
        self.calc_reg = calc_reg
        self.calc_seg = calc_seg
        self.calc_disp = calc_disp
        self.mode = mode

        if not os.path.exists(os.path.join(self.sv_dir, 'CSVs')):
            os.makedirs(os.path.join(self.sv_dir, 'CSVs'))

    def update_epoch(self, epoch):
        self.epoch = epoch
        self.df = pd.DataFrame()
    
    def load_avg_df(self):
        if os.path.exists(os.path.join(self.sv_dir, 'CSVs', 'trainMean_Results.csv')):
            self.avg_df = pd.read_csv(os.path.join(self.sv_dir, 'CSVs', 'trainMean_Results.csv'))
        else:
            print("Mean Results CSV not found!")

    def calculate_results(self, info, spacing, gt, rigid_msk = None, reg_result = None, seg_result = None, dvf = None):
        gt = gt.float().cpu().numpy()
        
        if reg_result != None:
            reg_result=reg_result.float().cpu().numpy()
            rigid_msk = rigid_msk.float().cpu().numpy()
            
            reg_dice = calculate_dice(gt, reg_result, len(self.class_list))
            reg_hd95 = calculate_hd_95(gt, reg_result, spacing, len(self.class_list))
            
            for i in range(len(self.class_list)):
                label = 'Reg_'+ self.class_list[i] + '_Dice'
                info[label] = reg_dice[i]

            for i in range(len(self.class_list)):
                label = 'Reg_'+ self.class_list[i] + '_HD95'
                info[label] = reg_hd95[i]

            if dvf != None:
                dvf=dvf.float().cpu().numpy()
                reg_disp = calculate_displacement(dvf, reg_result, len(self.class_list))
                for i in range(len(self.class_list)):
                    label = 'Reg_'+ self.class_list[i] + '_Displacement'
                    info[label] = reg_disp[i]
            rigid_dice = calculate_dice(gt, rigid_msk, len(self.class_list))
            rigid_hd95 = calculate_hd_95(gt, rigid_msk, spacing, len(self.class_list))

            for i in range(len(self.class_list)):
                label = 'Rigid_'+ self.class_list[i] + '_Dice'
                info[label] = rigid_dice[i]
            for i in range(len(self.class_list)):
                label = 'Rigid_'+ self.class_list[i] + '_HD95'
                info[label] = rigid_hd95[i]

            

        if seg_result != None:
            seg_result=seg_result.float().cpu().numpy()
            seg_dice = calculate_dice(gt, seg_result, len(self.class_list))
            seg_hd95 = calculate_hd_95(gt, seg_result, spacing, len(self.class_list))
        
            for i in range(len(self.class_list)):
                label = 'Seg_'+ self.class_list[i] + '_Dice'
                # print(len(seg_dice))
                # print(len(self.class_list))
                info[label] = seg_dice[i]
            for i in range(len(self.class_list)):
                label = 'Seg_'+ self.class_list[i] + '_HD95'
                info[label] = seg_hd95[i]
            
        self.df = pd.concat([self.df, pd.DataFrame([info])], ignore_index=True)
        if self.mode == 'train':
            self.df.to_csv(os.path.join(self.sv_dir, 'CSVs', str(self.epoch)+'_Results.csv'),index=False)
        else:
            self.df.to_csv(os.path.join(self.sv_dir, 'CSVs', self.mode+'_Results.csv'),index=False)
        

        return info

    def average_results(self):
        info = {}
        if self.mode == 'train':
            info['Epoch'] = self.epoch

        # Reg Dice
        if self.calc_reg:
            for i in range(len(self.class_list)):
                label = 'Reg_'+ self.class_list[i] + '_Dice'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))

            # Reg HD95
            for i in range(len(self.class_list)):
                label = 'Reg_'+ self.class_list[i] + '_HD95'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))

            # Rigid Dice
            for i in range(len(self.class_list)):
                label = 'Rigid_'+ self.class_list[i] + '_Dice'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))

            # Rigid HD95
            for i in range(len(self.class_list)):
                label = 'Rigid_'+ self.class_list[i] + '_HD95'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))

            if self.calc_disp:
                # Reg Displacement
                for i in range(len(self.class_list)):
                    label = 'Reg_'+ self.class_list[i] + '_Displacement'
                    info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))
        if self.calc_seg:
            # Seg Dice
            for i in range(len(self.class_list)):
                label = 'Seg_'+ self.class_list[i] + '_Dice'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))

            # Seg HD95
            for i in range(len(self.class_list)):
                label = 'Seg_'+ self.class_list[i] + '_HD95'
                info[label] = str(round(self.df[label].mean(),2))+' '+ u"\u00B1"+' '+str(round(self.df[label].std(),2))


        
        self.avg_df = pd.concat([self.avg_df, pd.DataFrame([info])], ignore_index=True)
        self.avg_df.to_csv(os.path.join(self.sv_dir, 'CSVs', self.mode+'Mean_Results.csv'),index=False)

        return info