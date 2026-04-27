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

from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode

from monai.networks.nets import EfficientNetBN

import pdb

device = torch.device("cuda")

def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):

    return Convolution(
        spatial_dims, in_channels, out_channels, strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True
    )


def get_upsample_layer(
    spatial_dims: int, in_channels: int, upsample_mode: Union[UpsampleMode, str] = "nontrainable", scale_factor: int = 2
):
    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )


class ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)


    def forward(self, x):

        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class SegResNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv,
                *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(blocks_down[i])],
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
            
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

    
class SegResNetVAE(SegResNet):
    def __init__(
        self,
        input_image_size: Sequence[int],
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[str, tuple] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=upsample_mode,
        )

        self.input_image_size = input_image_size
        self.smallest_filters = 16

        zoom = 2 ** (len(self.blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)

    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )
        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

    def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor):
        """
        Args:
            net_input: the original input of the network.
            vae_input: the input of VAE module, which is also the output of the network's encoder.
        """
        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean**2)

            x_vae = z_mean + z_sigma * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        vae_mse_loss = F.mse_loss(net_input, x_vae)
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

    def forward(self, x):
        net_input = x
        x, down_x = self.encode(x)

        down_x.reverse()

        vae_input = x
        x2 = self.decode(x, down_x)
        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
            return x2, vae_loss, x

        return x2, None, x
    
class MTLNet(nn.Module):
    def __init__(self):
        super(MTLNet, self).__init__()
        self.segmentation = SegResNetVAE(input_image_size=[512,384],init_filters=16,spatial_dims=2,in_channels=1,out_channels=1)
        self.classification = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=128, num_classes=2)
        
        checkpoint1 = torch.load("./weights/segresnet.pth",map_location="cuda")
        checkpoint2 = torch.load("./weights/efficientnet.pth",map_location="cuda")

        self.segmentation.load_state_dict(checkpoint1['model'])
        
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
        self.classification.load_state_dict(new_state_dict_class, strict=False)       
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        
        m_seg = self.segmentation(input)
        m_seg_features_encoder = m_seg[2]
        m_seg = m_seg[0]
        
        m_class = self.classification(m_seg_features_encoder)

        return m_seg, m_class