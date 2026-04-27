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

import warnings
warnings.filterwarnings("ignore")

class MTLNet(nn.Module):
    def __init__(self):
        super(MTLNet, self).__init__()
        
        self.segmentation = SegResNetVAE(input_image_size=[512,384],init_filters=16,spatial_dims=2,in_channels=1,out_channels=1,)
        self.classification = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=1, num_classes=2)
        
        checkpoint1 = torch.load("./weights/segresnet.pth",map_location="cuda")
        checkpoint2 = torch.load("./weights/efficientnet.pth",map_location="cuda")

        self.segmentation.load_state_dict(checkpoint1['model'])
        self.classification.load_state_dict(checkpoint2['model'])
        
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        
        m_seg = self.segmentation(input)
        m_seg = m_seg[0] 
        m_class = self.classification(input)

        return m_seg, m_class