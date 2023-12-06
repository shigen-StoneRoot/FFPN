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
