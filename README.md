# FFPN Repository

## Reference
If you take advantage of the data and paper in your research, please cite the following in your manuscript:

```
@article{SHI2025103442,
title = {Benefit from public unlabeled data: A Frangi filter-based pretraining network for 3D cerebrovascular segmentation},
journal = {Medical Image Analysis},
volume = {101},
pages = {103442},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2024.103442},
author = {Gen Shi and Hao Lu and Hui Hui and Jie Tian},
}
```

The whole data will be open source by Baidu Netdisk.

### Please be aware that we do not provide any raw image data. Instead, we will make the preprocessed data (the vessel masks and vessel-enhanced images), available in the .npy format as an open-source resource. So, you'll need to request the raw image data by yourself.

We provide detailed instructions for organizing and acquiring the public pretraining data (see in pretraining datasets folder).

Besides, the vessel masks of the three validation datasets have been open source in the validation dataset folder.

## Basic Dependency
 ```
python==3.8.0
einops==0.6.0
torch==1.11.0
monai==1.1.0
 ```

## Pretraining
After the dataset preprocessing, you can run the command to conduct the pretraining procedure:
```
python main_pretraining.py
```

## Pretraining Checkpoint

You can download the pretraining checkpoint at:

https://drive.google.com/file/d/1p6fHnP_bXuMjHcDBLqbDpCuKZG8-Osmk/view?usp=sharing

If you have any problem, please email me with this address: shigen@buaa.edu.cn
