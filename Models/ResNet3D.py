import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def build_norm_layer(cfg, num_features):
    layer_type = cfg.get('type', 'GN')
    requires_grad = cfg.get('requires_grad', True)
    if layer_type == 'BN3d':
        norm_layer = nn.BatchNorm3d(num_features)
    elif layer_type == 'GN':
        num_groups = cfg.get('num_groups', 32)
        norm_layer = nn.GroupNorm(num_groups, num_features)
    else:
        raise NotImplementedError(f'Norm layer {layer_type} is not implemented')

    for param in norm_layer.parameters():
        param.requires_grad = requires_grad
    return layer_type, norm_layer

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet3D(nn.Module):
    def __init__(self,
                 baseplanes=64,
                 block_strides=[1, 2, 2],
                 out_indices=(0, 1, 2),
                 input_channels=3,
                 shortcut_type='B',
                 norm_cfg=dict(type='GN', requires_grad=True),
                 widen_factor=1.0):
        super().__init__()

        depth_configs = (BasicBlock, [2, 2, 2])

        block, layers = depth_configs

        # channels for baseplanes * [1, 2, 4]
        block_inplanes = [int(baseplanes * widen_factor * (2 ** i)) for i in range(3)]
        self.in_planes = block_inplanes[0]
        self.out_indices = out_indices

        self.input_proj = nn.Sequential(
            #nn.Conv3d(input_channels, self.in_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv3d(input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, self.in_planes)[1],
            nn.ReLU(inplace=True)
        )

        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(self._make_layer(block, block_inplanes[i], layers[i],
                                                shortcut_type, block_strides[i], norm_cfg))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride, norm_cfg):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])
        layers = [block(self.in_planes, planes, stride, downsample, norm_cfg)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg))
        return nn.Sequential(*layers)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], device=out.device)
        return torch.cat([out, zero_pads], dim=1)

    def forward(self, x):
        x = self.input_proj(x)
        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                outputs.append(x)
        return outputs
    

class DenseConv3d(nn.Module):
    def __init__(self, input_channels=6, output_channels=64):
        super(DenseConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=input_channels*2, kernel_size=3, padding=1, stride=1) # k4,p1,s2
        self.conv2 = nn.Conv3d(in_channels=input_channels*2, out_channels=output_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv3d(in_channels=output_channels, out_channels=output_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(input_channels*2)
        self.bn2 = nn.BatchNorm3d(output_channels)
        self.bn3 = nn.BatchNorm3d(output_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        
        x = self.relu1(self.bn1(self.conv1(x)))
        
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.relu3(self.bn3(self.conv3(x)))
        
        return x