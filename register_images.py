import os
import os.path as osp

import ants
import nibabel as nib
from tempfile import mkstemp
import glob
import argparse

def to_nibabel(image):
    """Convert an ANTsImage to a Nibabel image"""
    fd, tmpfile = mkstemp(suffix=".nii.gz")
    try:
        image.to_filename(tmpfile)
        new_img = nib.load(tmpfile)
    finally:
        os.close(fd)
    return new_img

def process_and_save(planning_ct, planning_contours, cbct, contours, output_dir):
    """
    Process CBCT, contours, and planning CT contours, then save results.

    Args:
        planning_ct (ants.ANTsImage): Planning CT image.
        planning_contours (dict[str, ants.ANTsImage]): Dict of contour_name → planning contour (fixed).
        cbct (ants.ANTsImage): CBCT image to register.
        contours (dict[str, ants.ANTsImage]): Dict of contour_name → contour image (moving).
        output_dir (str): Directory where registered results will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Resample CBCT to planning CT space
    cbct_resampled = ants.resample_image(image=cbct, resample_params=planning_ct.spacing, use_voxels=False,
                                         interp_type=0)
    
    # Copy origin, direction, and spacing from one antsImage to another
    ants.core.ants_image.copy_image_info(planning_ct, cbct_resampled)

    # Register CBCT to planning CT
    # Reproduced from the paper wherein CBCT had a smaller FOV in comparison to CT and had to be aligned
    cbct_registered = ants.registration(fixed=planning_ct, moving=cbct_resampled, type_of_transform='Rigid')
    print(f'Shape of registered image: {cbct_registered["warpedmovout"].shape}')

    # Crop planning CT based on registered CBCT mask
    cropped_planning_ct = ants.crop_image(image=planning_ct, label_image=ants.get_mask(cbct_registered['warpedmovout']))
    print(f'Cropping planning CT: {cropped_planning_ct.shape}')
    
    cropped_cbct = ants.crop_image(image=cbct_registered["warpedmovout"], label_image=ants.get_mask(cbct_registered['warpedmovout']))
    print(f'Cropping CBCT: {cropped_cbct.shape}')

    # Save cropped results
    os.makedirs(osp.join(output_dir, "ct"), exist_ok=True)
    os.makedirs(osp.join(output_dir, "cbct"), exist_ok=True)
    nib.save(to_nibabel(cropped_planning_ct), osp.join(output_dir, "ct", "planct_image.nii.gz"))
    nib.save(to_nibabel(cropped_cbct), osp.join(output_dir, "cbct", "cbct_image.nii.gz"))

    # Process and save CBCT contours
    for contour_name, contour_image in contours.items():
         # Resample contour to planning CT space using nearest neighbor interpolation
        contour_resampled = ants.resample_image(image=contour_image, resample_params=planning_ct.spacing,
                                                use_voxels=False, interp_type=1)

        # Copy origin, direction, and spacing from one antsImage to another
        ants.core.ants_image.copy_image_info(planning_ct, contour_resampled)

        # Apply the same transformation as CBCT
        contour_registered = ants.apply_transforms(fixed=planning_ct, moving=contour_resampled,
                                                   transformlist=cbct_registered['fwdtransforms'], interpolator='nearestNeighbor')

        print(f'Shape of registered contour: {contour_registered["warpedmovout"].shape}')

        # Crop the registered contour
        contour_cropped = ants.crop_image(image=contour_registered,
                                          label_image=ants.get_mask(cbct_registered['warpedmovout']))
        print(f'Shape of cropped contour: {contour_cropped.shape}')
        
        # Save the processed contour
        nib.save(to_nibabel(contour_cropped),
                 osp.join(output_dir, 'cbct', f"cbct_{contour_name}.nii.gz"))

    # Process and save planning CT contours
    for contour_name, contour_image in planning_contours.items():
        # Crop the planning contour
        contour_cropped = ants.crop_image(image=contour_image,
                                          label_image=ants.get_mask(cbct_registered['warpedmovout']))
        # Save the processed planning contour
        nib.save(to_nibabel(contour_cropped),
                 osp.join(output_dir, 'ct', f"planct_{contour_name}.nii.gz"))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Rigid registration for CBCT -> CT')
    parser.add_argument('--path_images', default='dataset/image_inst1')
    args = parser.parse_args()
    
    ct_folder = osp.join(args.path_images, 'ct')
    patient_x_ct_path = glob.glob(osp.join(ct_folder, '*_image.nii.gz'))[0]
    planning_ct = ants.image_read(patient_x_ct_path)
    
    # Load planning CT contours
    planning_contours = {
        "label": ants.image_read(glob.glob(patient_x_ct_path.replace("_image", "_label"))[0])
    }

    output_dir = osp.join(args.path_images, "aligned_data")

    cbct_folder = osp.join(args.path_images, 'cbct')
    patient_x_cbcts = glob.glob(osp.join(cbct_folder, '*_image.nii.gz'))

    # # Process each cbct
    for idx in range(len(patient_x_cbcts)):
        cbct_path = patient_x_cbcts[idx]

        cbct = ants.image_read(cbct_path)
        cbct_contours = {
            "label": ants.image_read(cbct_path.replace("_image", "_label"))
        }

        process_and_save(planning_ct, planning_contours, cbct, cbct_contours, output_dir)

    print("Processing complete. Processed files saved in:", output_dir)
