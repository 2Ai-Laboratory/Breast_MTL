import os

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
from monai.handlers.utils import from_engine
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ROCAUCMetric #, MeanIoU
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import argparse
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import time
import glob
import numpy as np
from IPython.core.debugger import set_trace
import random
from monai.data.utils import pad_list_data_collate
import pdb
import torch.nn as nn
import nibabel as nib
import torch.nn.functional as F
import shutil

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc, average_precision_score

import re

import logging
logging.getLogger().setLevel(logging.ERROR)

class Train:
    def __init__(self,opt):

        modelph = opt.model
        folder = opt.folder
        root_dir = folder
        
        model_name = modelph
        model_name_save = model_name.split(".")[0]
        
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

        def assignLabel(label):
            if('benign' in label):
                gt = 0
            elif('malignant' in label):
                gt = 1
            else:
                gt = 2
            return gt

        data_dir = "./datasets/"
        
        test_labels_class = []
        
        test_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
        test_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsTs", "*.nii.gz")))
        test_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ]

        for name in test_images:
            label = re.search(r"\d\_(.*?)\.", name).group(1)
            num = assignLabel(label)
            test_labels_class.append(num)
            
        test_files = [
            {"image": image_name, "label1": label_name, "label2": label_name2}
            for image_name, label_name, label_name2 in zip(test_images, test_labels, test_labels_class)
        ]

        class_names = ['Benign', 'Malignant']
        num_class = len(class_names)
        
        set_determinism(seed=0)

        test_transforms = Compose(
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

        test_ds = CacheDataset(
            data=test_files, transform=test_transforms, cache_rate=1, num_workers=3)

        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

        device = torch.device("cuda")

        model = MTLNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        auc_metric = ROCAUCMetric()

        sig = nn.Sigmoid()

        checkpoint = torch.load(os.path.join(root_dir , model_name))
        model.load_state_dict(checkpoint['model'])

        k = 0
        y_true = []
        y_pred = []

        diceall = []
        prall = []
        reall = []
        accdall = []
        
        model.eval()
        with torch.no_grad():
            for test_data in test_loader:
                test_inputs, test_labels1, test_labels_class = (
                    test_data["image"].to(device),
                    test_data["label1"].to(device),
                    test_data["label2"].to(device),
                )
                
                dimsTest = np.shape(test_inputs)
                test_inputs = torch.reshape(test_inputs,[dimsTest[0],1,dimsTest[2],dimsTest[3]])
                test_labels1 = torch.reshape(test_labels1,[dimsTest[0],1,dimsTest[2],dimsTest[3]])
                
                outputs_seg, outputs_class = model(test_inputs)
                
                # Segmentation
                outputs_seg = sig(outputs_seg)
                outputs_seg = torch.where(outputs_seg>0.5, 1, 0)

                # Segmentation metrics
                for i in range(len(outputs_seg)):
                    outputs_seg_np = outputs_seg[i].cpu().numpy()
                    test_labels_np = test_labels1[i].cpu().numpy()
                    tp = np.logical_and(outputs_seg_np==1,test_labels_np==1).sum()
                    tn = np.logical_and(outputs_seg_np==0,test_labels_np==0).sum()
                    fn = np.logical_and(outputs_seg_np==0,test_labels_np==1).sum()
                    fp = np.logical_and(outputs_seg_np==1,test_labels_np==0).sum()
                    dicef = (2*tp)/(2*tp+fp+fn)
                    diceall.append(dicef)
                    prf = tp/(tp+fp)
                    prall.append(prf)
                    ref =  tp/(tp+fn)
                    reall.append(ref)
                    accdf =  (tp+tn)/(tp+tn+fp+fn)
                    accdall.append(accdf)

                # Classification
                pred = outputs_class.argmax(dim=1)
        
                for i in range(len(pred)):
                    y_true.append(test_labels_class[i].item())
                    y_pred.append(pred[i].item())
                    
                k = k + 1
                
        acc = accuracy_score(y_true, y_pred)
        roc = roc_curve(y_true, y_pred)
        aucroc = auc(roc[0], roc[1])

        plt.rcParams.update({'font.size':16})
        cm = confusion_matrix(y_true, y_pred)
        pdb.set_trace()
        cm_str = np.array2string(confusion_matrix(y_true, y_pred))
        cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)

        file = open(root_dir + "/test_" + model_name_save + ".txt","w")
        file.write("Classification Report: \n" + classification_report(y_true, y_pred, target_names=class_names, digits=4) + "\nConfusion Matrix: \n" + cm_str + "\nAccuracy: " + str(acc) + "\nAUC (excluding 'normal' class): " + str(aucroc))
        file.write('\n')
        file.write('Segmentation Report: \n')
        file.write('Mean: \n')
        file.write(f"dice: {np.mean(diceall)*100:.4f}")
        file.write('\n')
        file.write(f"pr: {np.mean(prall)*100:.4f}")
        file.write('\n')
        file.write(f"re: {np.mean(reall)*100:.4f}")
        file.write('\n')
        file.write(f"accd: {np.mean(accdall)*100:.4f}")
        file.write('\n')
        file.write('STD:\n')
        file.write(f"dice: {np.std(diceall)*100:.4f}")
        file.write('\n')
        file.write(f"pr: {np.std(prall)*100:.4f}")
        file.write('\n')
        file.write(f"re: {np.std(reall)*100:.4f}")
        file.write('\n')
        file.write(f"accd: {np.std(accdall)*100:.4f}")
        file.close()    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_mean_metrics_model.pth', help='trained model')
    parser.add_argument('--folder', type=str, default='', help='model folder')
    parser.add_argument('--type', type=str, default='model_MTL_Seg_Eff', help='MTL model type')
    
    opt = parser.parse_args()

    Train(opt)
