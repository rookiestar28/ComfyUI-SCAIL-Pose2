"""
modified torchvision EfficientNet with asymmetric padding (bottomright_stride).
"""
import collections
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import comfy.ops

# Helpers

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _fixed_padding(kernel_size, shifts=(0, 0)):
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    pad_v = [pad_beg - shifts[0], pad_end + shifts[0]]
    pad_h = [pad_beg - shifts[1], pad_end + shifts[1]]
    return nn.ZeroPad2d((*pad_h, *pad_v))


# Building blocks

class ConvBnAct(nn.Sequential):
    """Conv2d + BatchNorm2d + optional activation, using comfy.ops."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 groups=1, activation_layer=nn.SiLU, device=None, dtype=None,
                 operations=None):
        if operations is None:
            operations = comfy.ops.disable_weight_init
        if padding is None:
            padding = (kernel_size - 1) // 2
        layers = [
            operations.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False,
                              device=device, dtype=dtype),
            nn.BatchNorm2d(out_ch, eps=1e-3),
        ]
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SE(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, in_ch, squeeze_ch, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = operations.Conv2d(in_ch, squeeze_ch, 1, device=device, dtype=dtype)
        self.fc2 = operations.Conv2d(squeeze_ch, in_ch, 1, device=device, dtype=dtype)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.scale_activation(self.fc2(scale))
        return scale * x


# Block configs

@dataclass
class BlockConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    fused: bool  # True = FusedMBConv, False = MBConv
    bottomright_stride: bool = False

    @staticmethod
    def adjust_channels(channels, width_mult):
        return _make_divisible(channels * width_mult, 8)


# MBConv / FusedMBConv

class MBConv(nn.Module):
    def __init__(self, cnf: BlockConfig,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        expanded = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        layers = collections.OrderedDict()
        # expand
        if expanded != cnf.input_channels:
            layers[str(len(layers))] = ConvBnAct(
                cnf.input_channels, expanded, 1,
                device=device, dtype=dtype, operations=operations)
        # depthwise
        shifts = (1, 1) if cnf.bottomright_stride else (0, 0)
        layers['padding'] = _fixed_padding(cnf.kernel, shifts)
        layers[str(len(layers) - 1)] = ConvBnAct(
            expanded, expanded, cnf.kernel, stride=cnf.stride, padding=0,
            groups=expanded, device=device, dtype=dtype, operations=operations)
        # SE
        layers[str(len(layers) - 1)] = SE(
            expanded, max(1, cnf.input_channels // 4),
            device=device, dtype=dtype, operations=operations)
        # project (no activation)
        layers[str(len(layers) - 1)] = ConvBnAct(
            expanded, cnf.out_channels, 1, activation_layer=None,
            device=device, dtype=dtype, operations=operations)

        self.block = nn.Sequential(layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class FusedMBConv(nn.Module):
    def __init__(self, cnf: BlockConfig,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        expanded = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        layers = collections.OrderedDict()
        shifts = (1, 1) if cnf.bottomright_stride else (0, 0)
        layers['padding'] = _fixed_padding(cnf.kernel, shifts)

        if expanded != cnf.input_channels:
            # fused expand
            layers[str(len(layers) - 1)] = ConvBnAct(
                cnf.input_channels, expanded, cnf.kernel, stride=cnf.stride,
                padding=0, device=device, dtype=dtype, operations=operations)
            # project (no activation)
            layers[str(len(layers) - 1)] = ConvBnAct(
                expanded, cnf.out_channels, 1, activation_layer=None,
                device=device, dtype=dtype, operations=operations)
        else:
            layers[str(len(layers) - 1)] = ConvBnAct(
                cnf.input_channels, cnf.out_channels, cnf.kernel,
                stride=cnf.stride, padding=0,
                device=device, dtype=dtype, operations=operations)

        self.block = nn.Sequential(layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


# Preprocessing

class PreprocLayer(nn.Module):
    """Image normalization: (x - mean) / std."""
    def __init__(self, mean=0.5, std=0.5):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).reshape(1, 1, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).reshape(1, 1, 1, 1), persistent=False)

    def forward(self, x):
        mean = comfy.ops.cast_to_input(self.mean, x)
        std = comfy.ops.cast_to_input(self.std, x)
        return (x - mean) / std


# EfficientNetV2-L features builder

EFFNETV2_L_CONFIG = [
    #  expand, kernel, stride, in_ch, out_ch, layers, fused, bottomright
    BlockConfig(1, 3, 1,  32,  32,  4, fused=True),
    BlockConfig(4, 3, 2,  32,  64,  7, fused=True),
    BlockConfig(4, 3, 2,  64,  96,  7, fused=True),
    BlockConfig(4, 3, 2,  96, 192, 10, fused=False),
    BlockConfig(6, 3, 1, 192, 224, 19, fused=False),
    BlockConfig(6, 3, 2, 224, 384, 25, fused=False, bottomright_stride=True),
    BlockConfig(6, 3, 1, 384, 640,  7, fused=False),
]
EFFNETV2_L_LAST_CHANNEL = 1280


def build_backbone(device=None, dtype=None, operations=None):
    """Build EfficientNetV2-L backbone with comfy.ops.

    Returns Sequential(PreprocLayer, features) with 1280 output channels.
    """
    if operations is None:
        operations = comfy.ops.disable_weight_init

    setting = EFFNETV2_L_CONFIG

    layers = collections.OrderedDict()
    # stem
    layers['padding'] = _fixed_padding(3)
    layers[str(len(layers) - 1)] = ConvBnAct(
        3, setting[0].input_channels, 3, stride=2, padding=0,
        device=device, dtype=dtype, operations=operations)

    # stages
    for cnf in setting:
        stage = []
        for i in range(cnf.num_layers):
            bc = copy.copy(cnf)
            if i > 0:
                bc.input_channels = bc.out_channels
                bc.stride = 1
                bc.bottomright_stride = False
            block_cls = FusedMBConv if bc.fused else MBConv
            stage.append(block_cls(bc, device=device, dtype=dtype, operations=operations))
        layers[str(len(layers) - 1)] = nn.Sequential(*stage)

    # final conv
    layers[str(len(layers) - 1)] = ConvBnAct(
        setting[-1].out_channels, EFFNETV2_L_LAST_CHANNEL, 1,
        device=device, dtype=dtype, operations=operations)

    features = nn.Sequential(layers)
    backbone = nn.Sequential(PreprocLayer(0.5, 0.5), features)
    return backbone, EFFNETV2_L_LAST_CHANNEL
