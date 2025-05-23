#%%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize_image(img):
    img = img - np.min(img)
    img = img / np.max(img)
    return (img * 255).astype(np.uint8)

def save_all_segmentation_slices(image_path, gt_path, seg_path, output_dir='outputs', thickness=2):
    os.makedirs(output_dir, exist_ok=True)

    image = load_nifti(image_path)
    gt = load_nifti(gt_path)
    seg = load_nifti(seg_path)

    assert image.shape == gt.shape == seg.shape, f"Shape mismatch: image {image.shape}, gt {gt.shape}, seg {seg.shape}"

    depth = image.shape[2]

    for z in range(depth):
        img_slice = image[:, :, z]
        gt_edges = find_boundaries(gt[:, :, z], mode='inner')
        seg_edges = find_boundaries(seg[:, :, z], mode='inner')

        if thickness > 1:
            gt_edges = binary_dilation(gt_edges)
            seg_edges = binary_dilation(seg_edges)

        norm_img = normalize_image(img_slice)
        img_rgb = np.stack((norm_img,) * 3, axis=-1)

        img_rgb[gt_edges] = [255, 255, 0]  # yellow for GT
        img_rgb[seg_edges] = [255, 0, 0]   # red for SEG

        img_rgb = np.rot90(img_rgb)
        img_rgb = np.fliplr(img_rgb)

        plt.imsave(os.path.join(output_dir, f'slice_{z:03d}.png'), img_rgb)

# Example usage: we used image, groundtruth and segmentation in nifti format
save_all_segmentation_slices("image.nii", "gt.nii", "seg.nii", thickness=2)

# %%
