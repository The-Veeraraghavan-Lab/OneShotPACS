### Project Page: One-Shot PACS
### Patient specific Anatomic Context and Shape prior aware recurrent registration-segmentation of longitudinal thoracic cone beam CTs 

<img src="imgs/figure.png" width="1200px"/>

## Prerequisites
- Linux
- Python 3.9
- NVIDIA GPU with CUDA CuDNN

## To get started
- Clone this repository
- Download the corresponding weights from <a href="https://mskcc.box.com/s/x4ilt7xc69s47bu81zqynos39xj4r0zw">here</a> and save them to `saved_weights` folder inside `sv_dir`
- Install the requirements using `pip install -r requirements.txt`
- Run the script using
  ```bash
  python inference.py
  ——reg_model VoxelMorph -—seg_model CLSTM -—gpu 0 -—batch_size 1\
  —-sv_dir {savedir} --val_dir {valdir} --results_file {resultsfile} \
  --reg_weights saved_weights/0190_reg.pt \
  --seg_weights saved_weights/0190_seg.pt \
  --flown 8 -—data_dir {datadir} —-json_list {list of jsons}
  —-a_min -500 —-a_max 500 —-b_min 0.0 —-b_max 1.0 \
  —-space_x 1.5 —-space_y 1.5 —-space_z 1.5 \
  —-roi_x 256 —-roi_y 256 —-roi_z 64 \
  —-RandFlip_prob 0.45 —-RandRotate90d_prob 0.2 \
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
