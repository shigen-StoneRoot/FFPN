# FFPN Repository


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

## Reference
If you take advantage of the data and paper in your research, please cite the following in your manuscript:

```
@article{shi2023benefit,
  title={Benefit from public unlabeled data: A Frangi filtering-based pretraining network for 3D cerebrovascular segmentation},
  author={Shi, Gen and Lu, Hao and Hui, Hui and Tian, Jie},
  journal={arXiv preprint arXiv:2312.15273},
  year={2023}
}
```

If you have any problem, please email me with this address: shigen@buaa.edu.cn
