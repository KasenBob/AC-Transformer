import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from utils import *


class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    def __init__(self, p = None, kernel_size = 31, expansion_factor = 2, dropout_p = 0.1):
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2
        
        self.sequentia = nn.Sequential(
            #TODO:Rezero
            # nn.LayerNorm(p.d_model),
            #print_size(1),
            Transpose(shape=(1, 2)),
            #print_size(2),
            PointwiseConv1d(p.d_model, p.d_model * expansion_factor, stride=1, padding=0, bias=True),
            #print_size(3),
            GLU(dim=1),
            #print_size(4),
            DepthwiseConv1d(p.d_model, p.d_model, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            #print_size(5),
            nn.BatchNorm1d(p.d_model),
            #print_size(6),
            Swish(),
            #print_size(7),
            PointwiseConv1d(p.d_model, p.d_model, stride=1, padding=0, bias=True),
            #print_size(8),
            nn.Dropout(p=dropout_p),
            #print_size(9),
            Transpose(shape=(1, 2)),
            #print_size(10),
        )


    def forward(self, inputs):
        outputs = self.sequentia(inputs)
        return outputs