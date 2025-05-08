## pretraining datasets
Due to upload size limitations, we are unable to upload all the pretraining data (~200G) here.

All the preprocessed pretraining data (coarse segmentation and vessel-enhanced images in .npy format) have been open source at https://pan.baidu.com/s/11klozQG0UAawM_0TU8UdUw?pwd=gz11 

password：gz11 

Instead, you can re-preprocess the pretraining data by the following steps:

### (1) Download the image data

You can download the public MRA-TOF data at the following public websites:

IXI: http://brain-development.org/ixi-dataset/

OASIS3: https://www.oasis-brains.org/#data

TubeTK: https://public.kitware.com/Wiki/TubeTK/Data

ADAM: https://adam.isi.uu.nl/data/

BrainAneurysm: https://openneuro.org/datasets/ds003949/versions/1.0.1

The subjects used in this study have been stored in the ./subjects folder.

### (2) Organize the data according to the given example (TubeTK)

For each dataset, you should organize the structure as (take TubeTK as an example):
 ```
--TubeTK
----image
    subject1.nii.gz
    subject2.nii.gz
    .......
    subjectN.nii.gz
----hessian
----coarse_gt
 ```

### (3) Cropping the image data

The .pkl file contains the shape size after cropping. You can crop the data by the example code:

 ```
import nibabel as nib
import pickle
import os

def arr2nii(img_arr, origin_nii, output_path):
    affine, hdr = origin_nii.affine.copy(), origin_nii.header.copy()
    new_nii = nib.Nifti1Image(img_arr, affine, hdr)
    nib.save(new_nii, output_path)

shapes = pickle.load(open('TubeTK_shapes_pretraining.pkl', 'rb'))
sub = 'Normal001-MRA.nii.gz'
ori_nii_path = os.path.join(r'./TubeTK', 'image', sub)
tar_nii_path = os.path.join(r'./TubeTK', 'image', sub) # This will overwrite the original file.
nii = nib.load(ori_nii_path)
ori_img = nii.get_fdata()
ori_shape = shapes[sub][0]
x1, x2, y1, y2 = shapes[sub][1]
cropped_img = ori_img[x1:x2, y1:y2, :]
arr2nii(cropped_img, nii, tar_nii_path)
 ```
This step is optional. Please note we have conducted this operation on the provided data in Baidu Netdisk. 

### (4) Generate the hessian and coarse_gt

You have to modify the path in coarse_label.m file and run this file. After execution, please normalize the Hessian image to the range [0, 1], and binarize the coarse_gt data by setting all non-zero elements to 1.



After the preprocessing, all datasets should have the same file structure:
 ```
--TubeTK
----image
    subject1.nii.gz
    subject2.nii.gz
    .......
    subjectN.nii.gz
----hessian
    subject1.nii.gz
    subject2.nii.gz
    .......
    subjectN.nii.gz
----coarse_gt
    subject1.nii.gz
    subject2.nii.gz
    .......
    subjectN.nii.gz
 ```
