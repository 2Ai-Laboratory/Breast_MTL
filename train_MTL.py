from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandZoomd,
    RandGridDistortiond,
    Activations
)

from monai.metrics import DiceMetric, ROCAUCMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import torch.nn as nn
import os
import time
import glob
import numpy as np
from monai.data.utils import pad_list_data_collate
import nibabel as nib
import re
import argparse

import logging
logging.getLogger().setLevel(logging.ERROR)

class Train:
    def __init__(self, opt):
    
        type_net = opt.type
        
        if (type_net == "model_MTL_Seg_CB"):
            from models.model_MTL_Seg_CB import MTLNet
        elif (type_net == "model_MTL_Seg_Eff"):
            from models.model_MTL_Seg_Eff import MTLNet        
        elif (type_net == "model_MTL_Seg_Eff_C"):
            from models.model_MTL_Seg_Eff_C import MTLNet
        elif (type_net == "model_MTL_Seg_Eff_E"):
            from models.model_MTL_Seg_Eff_E import MTLNet
        elif (type_net == "model_MTL_Seg_Eff_EF"):
            from models.model_MTL_Seg_Eff_EF import MTLNet
        elif (type_net == "model_MTL_Seg_Eff_FF"):
            from models.model_MTL_Seg_Eff_FF import MTLNet
    
        data_dir = "./datasets/"
         
        train_labels_class = []
        val_labels_class = []

        def assignLabel(label):
            if('benign' in label):
                gt = 0
            elif('malignant' in label):
                gt = 1
            else:
                gt = 2
            return gt

        #Train
        train_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

        for name in train_images:
            label = re.search(r"\d\_(.*?)\.", name).group(1)
            num = assignLabel(label)
            train_labels_class.append(num)

        train_files = [
            {"image": image_name, "label1": label_name, "label2": label_name2}
            for image_name, label_name, label_name2 in zip(train_images, train_labels, train_labels_class)
        ]

        #Val
        val_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesVal", "*.nii.gz")))
        val_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsVal", "*.nii.gz")))

        for name in val_images:
            label = re.search(r"\d\_(.*?)\.", name).group(1)
            num = assignLabel(label)
            val_labels_class.append(num)

        val_files = [
            {"image": image_name, "label1": label_name, "label2": label_name2}
            for image_name, label_name, label_name2 in zip(val_images, val_labels, val_labels_class)
        ]


        class_names = ['Benign', 'Malignant']
        num_class = len(class_names)
        
        set_determinism(seed=0)
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label1"]),
                EnsureChannelFirstd(keys=["image", "label1"]),

                RandScaleIntensityd(keys="image", factors=0.5, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=60, prob=0.5),
                RandGaussianNoised(keys="image", prob=0.5, mean=30, std=5),
                RandGaussianSmoothd(keys="image", prob=0.25),
                RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.5,2)),
                RandFlipd(keys=["image", "label1"], prob=0.5, spatial_axis=0),
                RandZoomd(keys=["image", "label1"], prob=0.5, min_zoom=1, max_zoom=1.3, mode="nearest"),
                RandGridDistortiond(keys=["image", "label1"], prob=1, distort_limit=(-0.3,0.3), mode=["bilinear","nearest"]),

                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),

                EnsureTyped(keys=["image", "label1"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label1"]),
                EnsureChannelFirstd(keys=["image", "label1"]),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                EnsureTyped(keys=["image", "label1"]),
            ]
        )

        y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
        y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)]) 
        
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1, num_workers=0)

        train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=0)

        val_ds = CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=1, num_workers=0)

        val_loader = DataLoader(val_ds, batch_size=5, num_workers=0)
        
        device = torch.device("cuda")

        model = MTLNet().to(device)

        loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.5, lambda_ce=1)
        loss_function2 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.33,0.67]).to(device))
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        auc_metric = ROCAUCMetric()

        sig = nn.Sigmoid()
        
        parent_dir = "./results/"
        mes_ext = {1: 'Jan', 2 : 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',6:'Jun',7:'Jul', 8:'Aug',9:'Sep', 10: 'Oct', 11:'Nov', 12:'Dec'}
        month = time.strftime("%m")
        time_str = time.strftime("%d_%H-%M-%S")
        filename = mes_ext[int(month)]+ time_str + "_" + type_net;

        path = os.path.join(parent_dir, filename)
        os.mkdir(path)

        root_dir = path
        
        max_epochs = 5000
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        best_metric1 = -1
        best_metric_epoch1 = -1
        best_mean_metrics = -1
        epoch_loss_values = []
        epoch_loss_values_val = []

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            epoch_loss_val = 0
            step = 0
            step_val = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels1, labels2 = (
                    batch_data["image"].to(device),
                    batch_data["label1"].to(device),
                    batch_data["label2"].to(device),
                )

                dims = np.shape(inputs)
                inputs = torch.reshape(inputs,[dims[0],1,dims[2],dims[3]])
                labels1 = torch.reshape(labels1,[dims[0],1,dims[2],dims[3]])

                optimizer.zero_grad()

                outputs1, outputs2 = model(inputs)

                loss1 = loss_function(outputs1, labels1)
                loss2 = loss_function2(outputs2, labels2)

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"train_loss: {loss.item():.4f}")

            # Loss/train
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        step_val += 1
                        val_inputs, val_labels1, val_labels_class = (
                            val_data["image"].to(device),
                            val_data["label1"].to(device),
                            val_data["label2"].to(device),
                        )
                        dimsVal = np.shape(val_inputs)
                        val_inputs = torch.reshape(val_inputs,[dimsVal[0],1,dimsVal[2],dimsVal[3]])
                        val_labels1 = torch.reshape(val_labels1,[dimsVal[0],1,dimsVal[2],dimsVal[3]])

                        val_outputs1, val_outputs2 = model(val_inputs)

                        y_pred = torch.cat([y_pred, val_outputs2], dim=0)
                        y = torch.cat([y, val_labels_class], dim=0)

                        loss_val1 = loss_function(val_outputs1, val_labels1)
                        loss_val2 = loss_function2(val_outputs2, val_labels_class)

                        loss_val = loss_val1 + loss_val2
                        epoch_loss_val += loss_val.item()
                        val_outputs1 = sig(val_outputs1)
                        val_outputs1 = torch.where(val_outputs1>0.9, 1, 0)

                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs1, y=val_labels1)

                    # Loss/val
                    epoch_loss_val /= step_val
                    epoch_loss_values_val.append(epoch_loss_val)

                    # Dice/val
                    # aggregate the final mean dice result   
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    # AUC/val
                    y_onehot = [y_trans(i) for i in decollate_batch(y)]
                    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred.cpu())]
                    auc_metric(y_pred_act, y_onehot)
                    result = auc_metric.aggregate()
                    auc_metric.reset()
                    del y_pred_act, y_onehot

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save({'model': model.state_dict(),'epoch': epoch+1,'optimizer': optimizer.state_dict()}, os.path.join(
                            root_dir, "best_metric_dice_model.pth"))
                        print("saved new best metric DICE model")
                    
                    if result > best_metric1:
                        best_metric1 = result
                        best_metric_epoch1 = epoch + 1
                        torch.save({'model': model.state_dict(),'epoch': epoch+1,'optimizer': optimizer.state_dict()}, os.path.join(
                            root_dir, "best_metric_auc_model.pth"))
                        print("saved new best metric AUC model")

                    mean_metrics = (metric + result)/2

                    if mean_metrics > best_mean_metrics:
                        best_mean_metrics = mean_metrics
                        torch.save({'model': model.state_dict(),'epoch': epoch+1,'optimizer': optimizer.state_dict()}, os.path.join(
                            root_dir, "best_mean_metrics_model.pth"))
                        print("saved new best mean metrics model")     
                                            
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}; current AUC: {result:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                        f"\nbest AUC: {best_metric1:.4f} "
                        f"at epoch: {best_metric_epoch1}"
                    )
                    
            if epoch % 100 == 0:
                torch.save({'model': model.state_dict(),'epoch': epoch+1,'optimizer': optimizer.state_dict()}, os.path.join(root_dir, str(epoch) + "_final_metric_model.pth"))            
        
        torch.save({'model': model.state_dict(),'epoch': epoch+1,'optimizer': optimizer.state_dict()}, os.path.join(
            root_dir, "final_metric_model.pth"))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='model_MTL_Seg_Eff', help='MTL model type')
    
    opt = parser.parse_args()

    Train(opt)