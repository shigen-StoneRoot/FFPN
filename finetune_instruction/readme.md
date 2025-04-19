## VesselFormer Fine-tuning on nnUNet v1

Our fine-tuning is based on the **nnUNet v1** framework. For environment setup and basic usage of nnUNet, please refer to the repos: 

[https://github.com/zfdong-code/MNet](https://github.com/zfdong-code/MNet)

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)

### Step-by-step Instructions

1. **Set up the nnUNet v1 environment** as described in the above repository.

2. **Place the custom trainer and network files** into the corresponding nnUNet directories:

   - Copy `VesselTrainerV2.py` to:
     ```
     nnUNet/nnunet/training/network_training/
     ```
   - Copy `VesselFormer.py` to:
     ```
     nnUNet/nnunet/network_architecture/
     ```

3. **Edit the pre-trained weights path**:

   - Open `VesselTrainerV2.py`, and modify **line 139** to set the correct path to your pretrained model weights.

4. **Start training** with the following command:

   ```bash
   CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres VesselTrainerV2 <dataset_id> <fold_id>
