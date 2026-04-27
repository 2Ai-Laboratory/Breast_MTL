# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from monai.networks.nets import SegResNetVAE
from monai.networks.nets import EfficientNetBN

import matplotlib.pyplot as plt

device = torch.device("cuda")

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, drop_prob):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch, affine = True),
            #nn.Dropout(p=drop_prob),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine = True),
            #nn.Dropout(p=drop_prob),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class MTLNet(nn.Module):
    def __init__(self):
        super(MTLNet, self).__init__()
        
        self.segmentation = SegResNetVAE(input_image_size=[512,384],init_filters=16,spatial_dims=2,in_channels=17,out_channels=1,)
        self.classification = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=17, num_classes=2)
        
        checkpoint1 = torch.load("./weights/segresnet.pth",map_location="cuda")
        checkpoint2 = torch.load("./weights/efficientnet.pth",map_location="cuda")
        
        # Load the segmentation model's state_dict
        state_dict_seg = checkpoint1['model']
        current_model_dict_seg = self.segmentation.state_dict()

        # Match and load weights with size checks
        new_state_dict_seg = {}
        for k, v in state_dict_seg.items():  # Iterate over the checkpoint's state dict
            if k in current_model_dict_seg and v.shape == current_model_dict_seg[k].shape:
                new_state_dict_seg[k] = v  # Use the checkpoint weight
            else:
                new_state_dict_seg[k] = current_model_dict_seg[k]  # Retain the model's existing weight

        # Load the updated state dictionary
        self.segmentation.load_state_dict(new_state_dict_seg, strict=False)
        
        # Load the classification model's state_dict
        state_dict_class = checkpoint2['model']
        current_model_dict_class = self.classification.state_dict()

        # Match and load weights with size checks
        new_state_dict_class = {}
        for k, v in state_dict_class.items():  # Iterate over the checkpoint's state dict
            if k in current_model_dict_class and v.shape == current_model_dict_class[k].shape:
                new_state_dict_class[k] = v  # Use the checkpoint weight
            else:
                new_state_dict_class[k] = current_model_dict_class[k]  # Retain the model's existing weight

        # Load the updated state dictionary
        self.segmentation.load_state_dict(new_state_dict_seg, strict=False)
        self.classification.load_state_dict(new_state_dict_class, strict=False)
        
        self.dbconv1 =  double_conv(1, 8, 0)
        self.dbconv2 =  double_conv(8, 16, 0)
        
    def forward(self, input):
        
        features1 = self.dbconv1(input)
        features2 = self.dbconv2(features1)

        input_features = torch.cat((features2,input), 1)

        m_seg = self.segmentation(input_features)
        m_seg = m_seg[0]
        
        m_class = self.classification(input_features)

        return m_seg, m_class