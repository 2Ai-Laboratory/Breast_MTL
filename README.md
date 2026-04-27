# Evaluating Multi-Task Network Architectures for Simultaneous Breast Lesion Segmentation and Classification in Ultrasound Images

Breast lesion segmentation and classification in ultrasound (US) images are two essential tasks for computer-aided diagnosis of breast cancer. However, these tasks are still challenging, mainly due to the high variability of lesions and the poor image quality of US. Numerous deep learning methods have been proposed to assist physicians in performing breast lesion segmentation and classification. Considering that these tasks are related to common features, learning them jointly through multi-task learning (MTL) represents a viable approach to improve the performance of each individual task. In this work, we present and compare multiple MTL network configurations for the simultaneous segmentation and classification of breast lesions in ultrasound images. Building on two state-of-the-art architectures, namely SegResNet for segmentation and EfficientNet for classification, we designed and evaluated several combined configurations of these models arranged in different MTL schemes. These configurations explore various levels of feature sharing and integration between the tasks, aiming to identify the most effective architectural arrangement for joint lesion segmentation and malignancy classification in breast ultrasound. 

## MTL Networks

![Diagram](./images/mtl_networks.png)

## Main Dependencies

- torch 1.10.1 
- monai 0.8.1
- numpy 1.21.6
- nibabel 3.2.1

## Data

The code assumes that the dataset is organized with the following directory structure:

```
datasets/
├── imagesTr/
│   ├── X_Type.nii.gz
│   └── ...
├── labelsTr/
│   ├── X_Type.nii.gz
│   └── ...
├── imagesVal/
│   ├── X_Type.nii.gz
│   └── ...
├── labelsVal/
│   ├── X_Type.nii.gz
│   └── ...
├── imagesTs/
│   ├── X_Type.nii.gz
│   └── ...
└── labelsTs/
    ├── X_Type.nii.gz
    └── ...
```
The folders imagesTr, imagesVal, and imagesTs correspond to the training, validation, and testing sets, respectively. The files follow the naming convention X_Type.nii.gz, where:
- X is the image identifier (e.g., a numeric index),
- Type indicates the lesion classification (benign or malignant).
For example: 1_benign.nii.gz.

The corresponding labelsTr, labelsVal, and labelsTs folders contain the ground truth segmentation masks for each image, with matching filenames.

## Training and Inference
### Training example
`python train_MTL.py --type "model_MTL_Seg_CB"`

Note: The trained model weights (.pth files) are saved in the directory:

`./results/<date>_model_MTL_Seg_CB`

### Testing example
`python test_MTL.py --model "best_mean_metrics_model.pth" --folder "./results/<date>_model_MTL_Seg_CB" --type "model_MTL_Seg_CB"`
