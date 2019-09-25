'''
Customized version of pytorch resnet, alexnets.
'''

import numpy, torch, math, os
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet
from torchvision.models.alexnet import model_urls as alexnet_model_urls

class CustomResNet(nn.Module):
    '''
    Customizable ResNet, compatible with pytorch's resnet, but:
     * The top-level sequence of modules can be modified to add
       or remove or alter layers.
     * Extra outputs can be produced, to allow backprop and access
       to internal features.
     * Pooling is replaced by resizable GlobalAveragePooling so that
       any size can be input (e.g., any multiple of 32 pixels).
     * halfsize=True halves striding on the first pooling to
       set the default size to 112x112 instead of 224x224.
    '''
    def __init__(self, size=None, block=None, layers=None, num_classes=1000,
            extra_output=None, modify_sequence=None, halfsize=False):
        standard_sizes = {
            18: (resnet.BasicBlock, [2, 2, 2, 2]),
            34: (resnet.BasicBlock, [3, 4, 6, 3]),
            50: (resnet.Bottleneck, [3, 4, 6, 3]),
            101: (resnet.Bottleneck, [3, 4, 23, 3]),
            152: (resnet.Bottleneck, [3, 8, 36, 3])
        }
        assert (size in standard_sizes) == (block is None) == (layers is None)
        if size in standard_sizes:
            block, layers = standard_sizes[size]
        if modify_sequence is None:
            modify_sequence = lambda x: x
        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer # for recent resnet
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        sequence = modify_sequence([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2,
                padding=3, bias=False)),
            ('bn1', norm_layer(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, stride=1 if halfsize else 2,
                padding=1)),
            ('layer1', self._make_layer(block, 64, layers[0])),
            ('layer2', self._make_layer(block, 128, layers[1], stride=2)),
            ('layer3', self._make_layer(block, 256, layers[2], stride=2)),
            ('layer4', self._make_layer(block, 512, layers[3], stride=2)),
            ('avgpool', GlobalAveragePool2d()),
            ('fc', nn.Linear(512 * block.expansion, num_classes))
        ])
        super(CustomResNet, self).__init__()
        for name, layer in sequence:
            setattr(self, name, layer)
        self.extra_output = extra_output

    def _make_layer(self, block, channels, depth, stride=1):
        return resnet.ResNet._make_layer(self, block, channels, depth, stride)

    def forward(self, x):
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

class CustomAlexNet(nn.Module):
    '''
    Customizable AlexNet, compatible with pytorch's alexnet, but:
     * The top-level sequence of modules can be modified to add
       or remove or alter layers.
     * Extra outputs can be produced, to allow backprop and access
       to internal features.
     * halfsize=True halves striding on the first convolution to
       allow 119x119 images to be processed rather than 227x227 only.
    '''
    def __init__(self, channels=None, num_classes=1000,
            extra_output=None, modify_sequence=None, halfsize=False):
        if channels is None:
            channels = [3, 64, 192, 384, 256, 256, 4096, 4096]
        if modify_sequence is None:
            modify_sequence = lambda x: x
        sequence = modify_sequence([
            ('conv1', nn.Conv2d(channels[0], channels[1],
                kernel_size=11, stride=4, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=1 if halfsize else 2)),
            ('conv2', nn.Conv2d(channels[1], channels[2],
                kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(channels[2], channels[3],
                kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(channels[3], channels[4],
                kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(channels[4], channels[5],
                kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('flatten', Vectorize()),
            ('dropout6', nn.Dropout()),
            ('fc6', nn.Linear(channels[5] * 6 * 6, channels[6])),
            ('relu6', nn.ReLU(inplace=True)),
            ('dropout7', nn.Dropout()),
            ('fc7', nn.Linear(channels[6], channels[7])),
            ('relu7', nn.ReLU(inplace=True)),
            ('fc8', nn.Linear(channels[7], num_classes))
        ])
        super(CustomAlexNet, self).__init__()
        for name, layer in sequence:
            setattr(self, name, layer)
        self.extra_output = extra_output

    def forward(self, x):
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        '''
        Translates from pytorch's AlexNet parameter names
        into the custom parameter names.
        '''
        custom_names = [
            ('features.0.', 'conv1.'),
            ('features.3.', 'conv2.'),
            ('features.6.', 'conv3.'),
            ('features.8.', 'conv4.'),
            ('features.10.', 'conv5.'),
            ('classifier.1.', 'fc6.'),
            ('classifier.4.', 'fc7.'),
            ('classifier.6.', 'fc8.')
        ]
        custom_state_dict = {}
        for k, v in state_dict.items():
            for op, np in custom_names:
                if k.startswith(op):
                    k = np + k[len(op):]
                    break
            custom_state_dict[k] = v
        super(CustomAlexNet, self).load_state_dict(custom_state_dict, **kwargs)

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), int(numpy.prod(x.size()[1:])))
        return x

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()
    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

if __name__ == '__main__':
    import torch.utils.model_zoo as model_zoo
    # Verify that at the default settings, pytorch standard pretrained
    # models can be loaded into each of the custom nets.
    print('Loading alexnet')
    model = CustomAlexNet()
    model.load_state_dict(model_zoo.load_url(alexnet_model_urls['alexnet']))
    print('Loading resnet18')
    model = CustomResNet(18)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
    print('Loading resnet34')
    model = CustomResNet(34)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet34']))
    print('Loading resnet50')
    model = CustomResNet(50)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    print('Loading resnet101')
    model = CustomResNet(101)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
    print('Loading resnet152')
    model = CustomResNet(152)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet152']))

