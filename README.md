# [AAAI25] A Black-Box Evaluation Framework for Semantic Robustness in Bird’s Eye View Detection

## Preparation
### Dataset
The code is currently running on the [mini version](https://www.nuscenes.org/nuscenes#download:~:text=and%20log%20information.-,Mini,-Subset%20of%20trainval) of NuScenes dataset.
The structure of `/data` folder should be as follows.
```bash
/RobustBEV/data
├── can_bus -> /datasets/nuscenes_mini/can_bus/
├── nuscenes
│   ├── maps -> /datasets/nuscenes_mini/maps
│   ├── samples -> /datasets/nuscenes_mini/samples/
│   └── sweeps -> /datasets/nuscenes_mini/sweeps/
├── nuscenes_infos_temporal_train.pkl 
└── nuscenes_infos_temporal_val.pkl 
 # I am using soft links here, and '->' indicates where the files actually are.
```

### Python Environment
Please refer to the [UniAD project](https://github.com/OpenDriveLab/UniAD/blob/v2.0/docs/INSTALL.md)

### Pathes
There are a few pathes should be noticed. Taking DETR3D as an example.

> In `RobustBEV/zoo/DETR3D/projects/configs/detr3d/detr3d_res101_gridmask.py`, line 118, the `data_root` should be updated to where the dataset has been installed. Other models have a similar logic when loading the dataset.
```python
data_root = '/datasets/nuscenes_mini'
```

> Also, in `RobustBEV/zoo/DETR3D/projects/configs/detr3d/detr3d_res101_gridmask.py` and other configure files, the `__base__` path should be replaced by the global path of `mmdetection3d` on your machine. For example
```python
_base_ = [
    '~/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '~/mmdetection3d/configs/_base_/default_runtime.py'
]
```

### Model Weights
```bash
# Download zip dataset from Google Drive

# DETR
filename='detr3d_resnet101.pth'
fileid='1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O'

# BEVFORMER
https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth
https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth

# POLARFORMER
filename='polarformer_r101.pth'
fileid='1Jgh49QJXls6XP6OAGhm744JHCGb7dGpP'

# ORA3D
filename='ora3d_r101.pth'
fileid='1jft64_8BJv3JjNrITS-f64wYcb5j3mxF'

# PETR
filename='petr_vov_1600.pth'
fileid='1SV0_n0PhIraEXHJ1jIdMu3iMg9YZsm8c'
filename='petr_p4_r50.pth'
fileid='1eYymeIbS0ecHhQcB8XAFazFxLPm3wIHY'
filename='petr_vov_800.pth'
fileid='1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ'
```

## Model evaluation
The following commands should be able to run the evaluation procedure on models.
```bash
cd /home/fu/workspace/RobustBEV/zoo/{model_name}/
ln -s ../../data data # only need to run once
sh ./tools/run_eval.sh
```
Please check the `README.md` in each model folder for more details.

## Acknowledgement
Thanks to the great open-source projects:
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [RoboBEV](https://github.com/Daniel-xsy/RoboBEV)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [NuScenes](https://www.nuscenes.org)

and the authors of the following models:
 -  **[PolarFormer](https://arxiv.org/abs/2206.15398), AAAI 2023.** [**`[Code]`**](https://github.com/fudan-zvg/PolarFormer)
 -  **[ORA3D](https://arxiv.org/abs/2207.00865), BMVC 2022.** [**`[Code]`**](https://github.com/anonymous2776/ora3d)
 -  **[PETR](https://arxiv.org/abs/2203.05625), ECCV 2022.** [**`[Code]`**](https://github.com/megvii-research/PETR)
 -  **[BEVFormer](https://arxiv.org/abs/2203.17270), ECCV 2022.** [**`[Code]`**](https://github.com/fundamentalvision/BEVFormer)
 -  **[DETR3D](https://arxiv.org/abs/2110.06922), CoRL 2021.** [**`[Code]`**](https://github.com/WangYueFt/detr3d)

## Citation
If you find our method useful, please kindly cite our paper:
```
@inproceedings{fu2025blackbox,
    title = "A Black-Box Evaluation Framework for Semantic Robustness in Bird’s Eye View Detection",
    author="Fu Wang and Yanghao Zhang and Xiangyu Yin and Guangliang Cheng and Zeyu Fu and Xiaowei Huang and Wenjie Ruan",
    booktitle = "AAAI",
    year = "2025",
}
```