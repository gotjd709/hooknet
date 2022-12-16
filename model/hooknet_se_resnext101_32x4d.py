"""
Hooknet_SE-ResNeXt101_32x4d
description: The same model as the structure in the paper (https://arxiv.org/abs/2006.12230) with backbone SE-ResNeXt101_32x4d model (https://arxiv.org/pdf/1709.01507v4.pdf). 
Number of input image: 2 images with the same size and center but with different resolution (Target: high resolution, Context: low resolution)
size: (2, 3, 512, 512) -> (512, 512)

ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torchvision.transforms
import torchvision.models as models

"""

SE_ResNext101_32x4d models

"""

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


"""

Block models

"""

class Backbone_Block(nn.Module):
    
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()        
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        
        return x


class TransConv_Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return(x)

"""

HookNet

"""

class Hooknet(nn.Module):
    
    def __init__(self, class_num, hook_indice):
        super().__init__()

        self.hook_indice = hook_indice
        
        self.channel_list = [[3328,3200], [896,832], [448,416], [160,144]]
        self.resize_list = [[[24,40], [16,48]], [[48,80], [32,96]], [[96,160], [64,192]], [[192,320], [128,384]]]
        
        # (256x256) x 64
        self.layer1 = se_resnext101_32x4d().__dict__['_modules']['layer0'][:-1]
        # (128x128) x 256
        self.layer2 = nn.Sequential(
            se_resnext101_32x4d().__dict__['_modules']['layer0'][-1],
            se_resnext101_32x4d().__dict__['_modules']['layer1']
        )
        # (64x64) x 512
        self.layer3 = se_resnext101_32x4d().__dict__['_modules']['layer2']
        # (32x32) x 1024
        self.layer4 = se_resnext101_32x4d().__dict__['_modules']['layer3']
        # (16x16) x 2048
        self.layer5 = se_resnext101_32x4d().__dict__['_modules']['layer4']
        
        if self.hook_indice[0] == 0:
            try: 
                self.block52_1t = Backbone_Block(self.channel_list[0][self.hook_indice[1]], 256, 3, 1, 1)
            except:
                raise ValueError(f'hook_indice[1] must be 0 or 1')
        else:
            self.block52_1t = Backbone_Block(3072, 256, 3, 1, 1)
        self.block52_1c = Backbone_Block(3072, 256, 3, 1, 1)    
        self.block52_2 = Backbone_Block(256, 256, 3, 1, 1)

        if self.hook_indice[0] == 1:
            try:
                self.block42_1t = Backbone_Block(self.channel_list[1][self.hook_indice[1]-1], 128, 3, 1, 1)
            except:
                raise ValueError(f'hook_indice[1] must be 1 or 2')
        else:
            self.block42_1t = Backbone_Block(768, 128, 3, 1, 1)
        self.block42_1c = Backbone_Block(768, 128, 3, 1, 1)
        self.block42_2 = Backbone_Block(128, 128, 3, 1, 1)

        if self.hook_indice[0] == 2:
            try:
                self.block32_1t = Backbone_Block(self.channel_list[2][self.hook_indice[1]-2], 64, 3, 1, 1)
            except:
                raise ValueError(f'hook_indice[1] must be 2 or 3')
        else:
            self.block32_1t = Backbone_Block(384, 64, 3, 1, 1)
        self.block32_1c = Backbone_Block(384, 64, 3, 1, 1)
        self.block32_2 = Backbone_Block(64, 64, 3, 1, 1)

        if self.hook_indice[0] == 3:
            try: 
                self.block22_1t = Backbone_Block(self.channel_list[3][self.hook_indice[1]-3], 32, 3, 1, 1)
            except:
                raise ValueError(f'hook_indice[1] must be 3 or 4')
        else:
            self.block22_1t = Backbone_Block(128, 32, 3, 1, 1)
        self.block22_1c = Backbone_Block(128, 32, 3, 1, 1)
        self.block22_2 = Backbone_Block(32, 32, 3, 1, 1)

        if self.hook_indice[0] == 4:
            try: 
                self.block12_1t = Backbone_Block(48, 16, 3, 1, 1)
            except:
                raise ValueError(f'hook_indice[1] must be 4')
        else:
            self.block12_1t = Backbone_Block(32, 16, 3, 1, 1)
        self.block12_1c = Backbone_Block(32, 16, 3, 1, 1)
        self.block12_2 = Backbone_Block(16, 16, 3, 1, 1)

        
        self.output1 = nn.Conv2d(16, class_num, 3, 1, 1)
        self.output2 = nn.Softmax()

    def context(self, x):
        ## Context Branch

        # Contracting Path
        y2 = self.layer1(x)  # (256x256) x 64
        y3 = self.layer2(y2)  # (128x128) x 256
        y4 = self.layer3(y3)  # (64x64) x 512     
        y5 = self.layer4(y4)  # (32x32) x 1024
        y6 = self.layer5(y5)  # (16x16) x 2048

        # Expanding Path
        y52 = F.interpolate(y6, scale_factor=2, mode="nearest")
        y52_1 = self.block52_1c(torch.cat([y52, y5], dim=1))        
        y52_2 = self.block52_2(y52_1) # 256
        
        y42 = F.interpolate(y52_2, scale_factor=2, mode="nearest")
        y42_1 = self.block42_1c(torch.cat([y42, y4], dim=1))
        y42_2 = self.block42_2(y42_1) # 128
        
        y32 = F.interpolate(y42_2, scale_factor=2, mode="nearest")
        y32_1 = self.block32_1c(torch.cat([y32, y3], dim=1))
        y32_2 = self.block32_2(y32_1) # 64

        y22 = F.interpolate(y32_2, scale_factor=2, mode="nearest")
        y22_1 = self.block22_1c(torch.cat([y22, y2], dim=1))
        y22_2 = self.block22_2(y22_1) # 32 

        y12 = F.interpolate(y22_2, scale_factor=2, mode="nearest")
        y12_1 = self.block12_1c(y12)
        y12_2 = self.block12_2(y12_1) # 16
        
        output_y1 = self.output1(y12_2)
        output_y2 = self.output2(output_y1)

        if self.hook_indice[1] == 0:
            return y52_2[:,:,8:24,8:24], output_y2
        elif self.hook_indice[1] == 1:
            return y42_2[:,:,self.resize_list[0][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[0][(1-(self.hook_indice[1]-self.hook_indice[0]))][1], self.resize_list[0][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[0][(1-(self.hook_indice[1]-self.hook_indice[0]))][1]], output_y2
        elif self.hook_indice[1] == 2:
            return y32_2[:,:,self.resize_list[1][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[1][(1-(self.hook_indice[1]-self.hook_indice[0]))][1], self.resize_list[1][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[1][(1-(self.hook_indice[1]-self.hook_indice[0]))][1]], output_y2
        elif self.hook_indice[1] == 3:
            return y22_2[:,:,self.resize_list[2][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[2][(1-(self.hook_indice[1]-self.hook_indice[0]))][1], self.resize_list[2][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[2][(1-(self.hook_indice[1]-self.hook_indice[0]))][1]], output_y2
        elif self.hook_indice[1] == 4:
            return y12_2[:,:,self.resize_list[3][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[3][(1-(self.hook_indice[1]-self.hook_indice[0]))][1], self.resize_list[3][(1-(self.hook_indice[1]-self.hook_indice[0]))][0]:self.resize_list[3][(1-(self.hook_indice[1]-self.hook_indice[0]))][1]], output_y2

    
    def forward(self, x):
        ## Target Branch

        # Contracting Path
        x2 = self.layer1(x[:,0,...])  # (256x256) x 64
        x3 = self.layer2(x2)  # (128x128) x 256
        x4 = self.layer3(x3)  # (64x64) x 512
        x5 = self.layer4(x4)  # (32x32) x 1024
        x6 = self.layer5(x5)  # (16x16) x 2048

        ## Context Model
        y_hook, output_y2 = self.context(x[:,1,...])
        
        # Expanding Path
        if self.hook_indice[0] == 0:
            x6 = torch.cat([x6, y_hook], dim=1) # 2048+256, 2048+128
        x52 = F.interpolate(x6, scale_factor=2, mode="nearest")
        x52_1 = self.block52_1t(torch.cat([x52, x5], dim=1)) # x+1024        
        x52_2 = self.block52_2(x52_1)

        if self.hook_indice[0] == 1:
            x52_2 = torch.cat([x52_2, y_hook], dim=1) # 256 + 128, 256 + 64
        x42 = F.interpolate(x52_2, scale_factor=2, mode="nearest")
        x42_1 = self.block42_1t(torch.cat([x42, x4], dim=1)) # x + 512
        x42_2 = self.block42_2(x42_1)

        if self.hook_indice[0] == 2:
            x42_2 = torch.cat([x42_2, y_hook], dim=1) # 128 + 64, 128 + 32           
        x32 = F.interpolate(x42_2, scale_factor=2, mode="nearest")
        x32_1 = self.block32_1t(torch.cat([x32, x3], dim=1)) # x + 256
        x32_2 = self.block32_2(x32_1)

        if self.hook_indice[0] == 3:
            x32_2 = torch.cat([x32_2, y_hook], dim=1) # 64 + 32, # 64 + 16
        x22 = F.interpolate(x32_2, scale_factor=2, mode="nearest")
        x22_1 = self.block22_1t(torch.cat([x22, x2], dim=1)) # x + 64
        x22_2 = self.block22_2(x22_1)

        if self.hook_indice[0] == 4:
            x22_2 = torch.cat([x22_2,  y_hook], dim=1) # 32 + 16
        x12 = F.interpolate(x22_2, scale_factor=2, mode="nearest")
        x12_1 = self.block12_1t(x12)
        x12_2 = self.block12_2(x12_1)
        
        output_x1 = self.output1(x12_2)
        output_x2 = self.output2(output_x1)
        output = torch.stack((output_x2, output_y2), dim=1)
        return output

def hooknet_se_resnext101_32x4d(class_num, hook_indice):
    return Hooknet(class_num, hook_indice) 