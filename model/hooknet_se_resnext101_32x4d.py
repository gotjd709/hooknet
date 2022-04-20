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
from torch.utils import model_zoo

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

class Identical_Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.ident = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.ident(x)
        return(x)

"""

HookNet

"""


class Hooknet(nn.Module):
    def __init__(self, class_num, hook_indice=[4,2]):
        super().__init__()   

        # Contracting Path 
        self.hook_indice = hook_indice
        self.layer00 = se_resnext101_32x4d().__dict__['_modules']['layer0'][:-1]
        self.layer01 = se_resnext101_32x4d().__dict__['_modules']['layer0'][-1]
        self.layer10 = se_resnext101_32x4d().__dict__['_modules']['layer1'][0]
        self.layer11 = se_resnext101_32x4d().__dict__['_modules']['layer1'][1:]
        self.layer20 = se_resnext101_32x4d().__dict__['_modules']['layer2'][0]
        self.layer21 = se_resnext101_32x4d().__dict__['_modules']['layer2'][1:]
        self.layer30 = se_resnext101_32x4d().__dict__['_modules']['layer3'][0]
        self.layer31 = se_resnext101_32x4d().__dict__['_modules']['layer3'][1:]
        self.layer40 = se_resnext101_32x4d().__dict__['_modules']['layer4'][0]
        self.layer41 = se_resnext101_32x4d().__dict__['_modules']['layer4'][1:]

        # Expanding Path
        self.cont12_1_2 = nn.ConvTranspose2d(2048, 256, 2, 2, 0)
        self.cont12_1_1 = nn.ConvTranspose2d(2048, 256, 2, 2, 0)
        self.block12_2 = Backbone_Block(256, 256, 3, 1, 1)
        self.block12_3 = Backbone_Block(1280, 256, 3, 1, 1)
        self.block12_4 = Backbone_Block(256, 256, 3, 1, 1)

        if self.hook_indice[0] == 4:
            if self.hook_indice[1] == 1:
                self.cont22_1_2 = nn.ConvTranspose2d(320, 256, 2, 2, 0)
            elif self.hook_indice[1] == 2:
                self.cont22_1_2 = nn.ConvTranspose2d(384, 256, 2, 2, 0)
            elif self.hook_indice[1] == 3:
                self.cont22_1_2 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont22_1_2 = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.cont22_1_1 = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.block22_2 = Backbone_Block(256, 256, 3, 1, 1)
        self.block22_3 = Backbone_Block(768, 256, 3, 1, 1)
        self.block22_4 = Backbone_Block(256, 256, 3, 1, 1)

        if self.hook_indice[0] == 3:
            if self.hook_indice[1] == 1:
                self.cont32_1_2 = nn.ConvTranspose2d(320, 128, 2, 2, 0)
            elif self.hook_indice[1] == 2:
                self.cont32_1_2 = nn.ConvTranspose2d(384, 128, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont32_1_2 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.cont32_1_1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.block32_2 = Backbone_Block(128, 128, 3, 1, 1)
        self.block32_3 = Backbone_Block(384, 128, 3, 1, 1)
        self.block32_4 = Backbone_Block(128, 128, 3, 1, 1)

        if self.hook_indice[0] == 2:
            if self.hook_indice[1] == 1:
                self.cont42_1_2 = nn.ConvTranspose2d(192, 64, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont42_1_2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.cont42_1_1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.block42_2 = Backbone_Block(64, 64, 3, 1, 1)
        self.block42_3 = Backbone_Block(128, 64, 3, 1, 1)
        self.block42_4 = Backbone_Block(64, 64, 3, 1, 1)

        self.cont52_1_1 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.cont52_1_2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.block52_2 = Backbone_Block(32, 32, 3, 1, 1)
        self.block52_3 = Backbone_Block(32, 16, 3, 1, 1)
        self.block52_4 = Backbone_Block(16, 16, 3, 1, 1)
        
        self.output1 = nn.Conv2d(16, class_num, 3, 1, 1)
        self.output2 = nn.Softmax()
        
    def context(self, x):
        # Contracting Path
        x00 = self.layer00(x[:,1,...]) # 256
        x01 = self.layer01(x00) # 128
        x10 = self.layer10(x01) 
        x11 = self.layer11(x10)
        x20 = self.layer20(x11) # 64
        x21 = self.layer21(x20)
        x30 = self.layer30(x21) # 32
        x31 = self.layer31(x30) 
        x40 = self.layer40(x31) # 16
        x41 = self.layer41(x40)

        # Expanding Path
        x12_1_1 = self.cont12_1_1(x41)
        x12_2 = self.block12_2(x12_1_1)        
        x12_3 = self.block12_3(torch.cat([x12_2, x31], dim=1))
        x12_4 = self.block12_4(x12_3)
        
        x22_1_1 = self.cont22_1_1(x12_4)
        x22_2 = self.block22_2(x22_1_1)
        x22_3 = self.block22_3(torch.cat([x22_2, x21], dim=1))
        x22_4 = self.block22_4(x22_3)
        
        if self.hook_indice[1] == 3:
            return x22_4
        x32_1_1 = self.cont32_1_1(x22_4)
        x32_2 = self.block32_2(x32_1_1)
        x32_3 = self.block32_3(torch.cat([x32_2, x11], dim=1))
        x32_4 = self.block32_4(x32_3)
        
        if self.hook_indice[1] == 2:
            return x32_4
        x42_1_1 = self.cont42_1_1(x32_4)
        x42_2 = self.block42_2(x42_1_1)
        x42_3 = self.block42_3(torch.cat([x42_2, x00], dim=1))
        x42_4 = self.block42_4(x42_3)

        if self.hook_indice[1] == 1:
            return x42_4

    
    def forward(self, x):
        # Contracting Path
        x00 = self.layer00(x[:,0,...]) # 256
        x01 = self.layer01(x00) # 128
        x10 = self.layer10(x01) 
        x11 = self.layer11(x10)
        x20 = self.layer20(x11) # 64
        x21 = self.layer21(x20)
        x30 = self.layer30(x21) # 32
        x31 = self.layer31(x30) 
        x40 = self.layer40(x31) # 16
        x41 = self.layer41(x40)

        # Expanding Path
        x12_1_2 = self.cont12_1_2(x41)
        x12_2 = self.block12_2(x12_1_2)        
        x12_3 = self.block12_3(torch.cat([x12_2, x31], dim=1))
        x12_4 = self.block12_4(x12_3)

        if self.hook_indice[0] == 4:
            y = self.context(x)
            x12_4 = torch.cat([x12_4, y[:, :, y.shape[2]//2-16:y.shape[2]//2+16, y.shape[3]//2-16:y.shape[3]//2+16]], dim=1)
        x22_1_2 = self.cont22_1_2(x12_4)
        x22_2 = self.block22_2(x22_1_2)
        x22_3 = self.block22_3(torch.cat([x22_2, x21], dim=1))
        x22_4 = self.block22_4(x22_3)

        if self.hook_indice[0] == 3:
            y = self.context(x)
            x22_4 = torch.cat([x22_4, y[:, :, (y.shape[2]//2)-32:(y.shape[2]//2)+32, (y.shape[3]//2)-32:(y.shape[3]//2)+32]], dim=1)
        x32_1_2 = self.cont32_1_2(x22_4)
        x32_2 = self.block32_2(x32_1_2)
        x32_3 = self.block32_3(torch.cat([x32_2, x11], dim=1))
        x32_4 = self.block32_4(x32_3)
        
        if self.hook_indice[0] == 2:
            y = self.context(x)
            x32_4 = torch.cat([x32_4, y[:, :, y.shape[2]//2-64:y.shape[2]//2+64, y.shape[3]//2-64:y.shape[3]//2+64]], dim=1)
        x42_1_2 = self.cont42_1_2(x32_4)
        x42_2 = self.block42_2(x42_1_2)
        x42_3 = self.block42_3(torch.cat([x42_2, x00], dim=1))
        x42_4 = self.block42_4(x42_3)
        
        x52_1_1 = self.cont52_1_1(x42_4)
        x52_2 = self.block52_2(x52_1_1)
        x52_3 = self.block52_3(x52_2)
        x52_4 = self.block52_4(x52_3)    

        output1 = self.output1(x52_4)
        output2 = self.output2(output1)        
        return output2

def hooknet_se_resnext101_32x4d(class_num, hook_indice):
    return Hooknet(class_num, hook_indice) 
