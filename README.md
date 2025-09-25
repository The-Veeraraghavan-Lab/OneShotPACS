# One-Shot PACS
### Patient specific Anatomic Context and Shape prior aware recurrent registration-segmentation of longitudinal thoracic cone beam CTs 

<img src="imgs/figure.png" width="100%"/>

---

## 📌 Overview
This repository provides the official implementation of **One-Shot PACS** [[Jiang & Veeraraghavan, *IEEE TMI 2022*](https://arxiv.org/pdf/2201.11000)], a recurrent registration–segmentation framework for longitudinal thoracic CBCTs with patient-specific anatomical context and shape priors.

---

## Prerequisites
- Linux
- Python 3.9
- NVIDIA GPU with CUDA CuDNN (CUDA 11.8 or higher)

---
## To get started
- Clone this repository
- Download the corresponding weights from <a href="https://mskcc.box.com/s/bzdu41q3obdy0gb4ywf45o49w03ekdpg">here</a> and save them to `saved_weights` folder inside `sv_dir`
- Install pytorch (our GPU is at CUDA 11.8, so we use the following command `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`)
- Install the requirements using `pip install -r requirements.txt`
- The data should be present in the `dataset` folder, with the json organizing each instance under `validation` key as the following template:
  - datasets
    - imagefiles
      - image1
        - ct
          - image
          - label
        - cbct
          - image(s)
          - label(s)
      - image2
      - image3
      - ...
  
- Note, the set of codes naturally assume that the name of the image is the last piece of information in the json file. If this is not the case, edit <a href="https://github.com/The-Veeraraghavan-Lab/OneShotPACS/blob/4d3de5038a1151c24b0d9818458fe7e399cc435d/inference.py#L233">this line</a> under `inference.py`
- In addition, every image scan should end with `_image` and every manual delineation/auto-segmentation scan should end with `_label`. This can be changed for the registration code `register_images.py`, and for the rest of the workflow by simply adding the correct correspondence in the json file.
- Run the registration script using
  ```bash
  python register_images.py
  --path_images {path to image directory (image1, image2, ...)}
  ```
  This will save the pre-processed registered data in the same image directory under the `aligned_data` folder. Note, this has already been run for the set of images given as examples in this page.
- Run the script using
  ```bash
  python inference.py
  ——reg_model VoxelMorph -—seg_model CLSTM -—gpu {gpu_id} -—batch_size 1 \
  —-sv_dir {savedir} --val_dir {name of directory to save results under (stored under sv_dir} \
  --reg_weights saved_weights/{weightfilename for registration} \
  --seg_weights saved_weights/{weightfilename for segmentation} \
  --flown 8 -—data_dir dataset —-json_list {name of json file containing the list of IDs} \
  —-a_min -500 —-a_max 500 —-b_min 0.0 —-b_max 1.0 \
  —-space_x 1.5 —-space_y 1.5 —-space_z 1.5 \
  —-roi_x 256 —-roi_y 256 —-roi_z 64 \
  —-RandFlipd_prob 0.45 —-RandRotate90d_prob 0.2 \
  —-RandShiftIntensityd_prob 0.1 —-workers 1
  ```
- For example, given the data used as example in this repository, the following python code should produce the CSV file located <a href="https://mskcc.box.com/s/x4ilt7xc69s47bu81zqynos39xj4r0zw">here</a> under `sv_dir/inference_190/CSVs/test_Results.csv`
  ```bash
  python inference.py
  ——reg_model VoxelMorph -—seg_model CLSTM -—gpu 0 -—batch_size 1 \
  —-sv_dir sv_dir --val_dir inference_190 \
  --reg_weights saved_weights/0190_reg.pt  \
  --seg_weights saved_weights/0190_seg.pt  \
  --flown 8 -—data_dir dataset —-json_list example.json \
  —-a_min -500 —-a_max 500 —-b_min 0.0 —-b_max 1.0 \
  —-space_x 1.5 —-space_y 1.5 —-space_z 1.5 \
  —-roi_x 256 —-roi_y 256 —-roi_z 64 \
  —-RandFlipd_prob 0.45 —-RandRotate90d_prob 0.2 \
  —-RandShiftIntensityd_prob 0.1 —-workers 1
  ```
  
## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/pdf/2201.11000">One shot PACS</a>:

```
@article{jiang2022one,
  title={One shot PACS: Patient specific Anatomic Context and Shape prior aware recurrent registration-segmentation of longitudinal thoracic cone beam CTs},
  author={Jiang, Jue and Veeraraghavan, Harini},
  journal={IEEE transactions on medical imaging},
  volume={41},
  number={8},
  pages={2021--2032},
  year={2022},
  publisher={IEEE}
}
```
