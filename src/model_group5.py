from __future__ import annotations
from typing import Sequence, Union
from escnn import nn as enn
from escnn.nn import *
import escnn 

import torch
from torch import nn
import numpy as np
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep

# from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class AdaptedViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    # Set the default parameters to the defaults given in the Segvol/train.py scripts
    def __init__(
        self,
        in_channels: int = 1,
        img_size: Union[Sequence[int], int] = (32, 256, 256),
        patch_size: Union[Sequence[int], int] = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        
        self.patch_embedding_equiv = SO3SteerablePatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=8,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # Create adapter module that concatenates two embeddings
        self.adapter = Adapter(
            input_size=hidden_size,
            hidden_size=1000,
            output_size=hidden_size
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

    def forward(self, x):
        emb1 = self.patch_embedding_equiv(x)
        emb2 = self.patch_embedding(x)
        print(emb1.shape)
        print(emb2.shape)
        x = torch.cat((emb1, emb2), dim=1)
        print(x.shape)
        x = self.adapter(x)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)

        return x, hidden_states_out
    
class Adapter(nn.Module):
    """ 
        Maybe worth to consider using a gate instead of linear layers to mix the embeddings. 
    """
    def __init__(
            self,
            input_size:int,
            hidden_size:int,
            output_size:int
    ):
        super().__init__()
        self.adapter_net = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)]
        )
    
    def forward(self, x):
        for layer in self.adapter_net:
            x = layer(x)
        return x

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SO3SteerablePatchEmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        # patch_size: Union[Sequence[int], int],
        patch_size: int,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        # The model will be equivariant under planer rotations
        self.r3_act = escnn.gspaces.rot3dOnR3()

        # The group SO(3)
        self.Group = self.r3_act.fibergroup

        # The input field is a scalar field, because we are dealing with 3D-gray images (CT scans)
        in_type = enn.FieldType(self.r3_act, [self.r3_act.trivial_repr])
        self.input_type = in_type
        # The output is still a scalar, but we use #hidden_size channels to create the embeddings for the ViT
        out_type = enn.FieldType(self.r3_act, hidden_size*[self.r3_act.trivial_repr])
        
        # The 3D group equivariant convolution, that performs one pass on each block (similar to normal ViT)
        self.patch_embeddings = enn.R3Conv(in_type, out_type, kernel_size=8, stride=8)
        init.generalized_he_init(self.patch_embeddings.weights.data, self.patch_embeddings.basisexpansion, cache=True)

        # Create the learnable position embeddings
        self.position_embeddings = torch.nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        torch.nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.patch_embeddings(x).tensor
        
        # Flatten all spatial dimensions into one, and add the learnable encoding
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return embeddings

    def check_equivariance(self):
        with torch.no_grad():
            x = torch.randn(3, 1, 32, 32 , 32)
            
            # the volumes rotated by 90 degrees in the ZY plane (i.e. around the X axis)
            x_x90 = x.rot90(1, (2, 3))
            x_x90 = enn.GeometricTensor(x_x90, self.input_type)
            yx = self.patch_embeddings(x_x90).tensor

            # the volumes rotated by 90 degrees in the XZ plane (i.e. around the Y axis)
            x_y90 = x.rot90(1, (2, 4))
            x_y90 = enn.GeometricTensor(x_y90, self.input_type)
            yy = self.patch_embeddings(x_y90).tensor

            # the volumes rotated by 90 degrees in the YX plane (i.e. around the Z axis)
            x_z90 = x.rot90(1, (3, 4))
            x_z90 = enn.GeometricTensor(x_z90, self.input_type)
            yz = self.patch_embeddings(x_z90).tensor

            # the volumes rotated by 180 degrees in the XZ plane (i.e. around the Y axis)
            x_y180 = x.rot90(2, (2, 4))
            x_y180 = enn.GeometricTensor(x_y180, self.input_type)
            yy180 = self.patch_embeddings(x_y180).tensor

            x = enn.GeometricTensor(x, self.input_type)
            y = self.patch_embeddings(x).tensor

            # Rotate the outputs back to the original orientation, and check whether output matches with y.
            print('TESTING INVARIANCE:                     ')
            print('90 degrees ROTATIONS around X axis:  ' + ('Equiv' if torch.allclose(y, yx.rot90(3, (2,3)), atol=1e-3) else 'False'))
            print('90 degrees ROTATIONS around Y axis:  ' + ('Equiv' if torch.allclose(y, yy.rot90(3, (2,4)), atol=1e-3) else 'False'))
            print('90 degrees ROTATIONS around Z axis:  ' + ('Equiv' if torch.allclose(y, yz.rot90(3, (3,4)), atol=1e-3) else 'False'))
            print('180 degrees ROTATIONS around Y axis: ' + ('Equiv' if torch.allclose(y, yy180.rot90(2, (2,4)), atol=1e-4) else 'False'))


