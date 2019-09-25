import torch
import torch.nn as nn
from collections import OrderedDict


def print_network(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('Total number of parameters: {:3.3f} M'.format(num_params / 1e6))


def split_model(model, layerID):
    model1 = nn.Sequential()
    model2 = nn.Sequential()
    for i, layer in enumerate(list(model.features)):
        name = "layer_" + str(i)
        if i <= layerID:
            model1.add_module(name, layer)
        else:
            model2.add_module(name, layer)
    model2.add_module('output', model.output)
    return model1, model2

###############################################################################
# Functions
###############################################################################


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, size):
        super(WScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn([1]))
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])

        return x


class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, no_pixel=False, no_wscale=False):
        super(NormConvBlock, self).__init__()
        self.norm = None
        self.wscale = None
        if not no_pixel:
            self.norm = PixelNormLayer()
        if not no_wscale:
            self.wscale = WScaleLayer(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=no_wscale)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.conv(x)
        if self.wscale is not None:
            x = self.wscale(x)
        x = self.relu(x)
        return x


class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, no_pixel=False, no_wscale=False):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = None
        self.wscale = None
        if not no_pixel:
            self.norm = PixelNormLayer()
        if not no_wscale:
            self.wscale = WScaleLayer(out_channels)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=no_wscale)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        if self.wscale is not None:
            x = self.wscale(x)
        x = self.relu(x)
        return x


class G128_pixelwisenorm(nn.Module):
    def __init__(self):
        super(G128_pixelwisenorm, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
            NormConvBlock(32, 32, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class G128_equallr(nn.Module):
    def __init__(self):
        super(G128_equallr, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3, no_pixel=False),
            NormConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True),
            NormConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1, no_pixel=True),
            NormConvBlock(256, 256, kernel_size=3, padding=1, no_pixel=True),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1, no_pixel=True),
            NormConvBlock(128, 128, kernel_size=3, padding=1, no_pixel=True),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1, no_pixel=True),
            NormConvBlock(64, 64, kernel_size=3, padding=1, no_pixel=True),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1, no_pixel=True),
            NormConvBlock(32, 32, kernel_size=3, padding=1, no_pixel=True))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class G128_minibatch_disc(nn.Module):
    def __init__(self):
        super(G128_minibatch_disc, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3, no_pixel=False, no_wscale=True),
            NormConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormConvBlock(512, 512, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormConvBlock(256, 256, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormConvBlock(128, 128, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormConvBlock(64, 64, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1, no_pixel=True, no_wscale=True),
            NormConvBlock(32, 32, kernel_size=3, padding=1, no_pixel=True, no_wscale=True))

        self.output = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=True)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class NormSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormSimpleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)  # LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class NormUpscaleSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleSimpleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)  # nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class G128_simple(nn.Module):
    def __init__(self):
        super(G128_simple, self).__init__()

        self.features = nn.Sequential(
            NormSimpleBlock(128, 512, kernel_size=4, padding=3),
            NormSimpleBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleSimpleBlock(512, 512, kernel_size=3, padding=1),
            NormSimpleBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleSimpleBlock(512, 256, kernel_size=3, padding=1),
            NormSimpleBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleSimpleBlock(256, 128, kernel_size=3, padding=1),
            NormSimpleBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleSimpleBlock(128, 64, kernel_size=3, padding=1),
            NormSimpleBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleSimpleBlock(64, 32, kernel_size=3, padding=1),
            NormSimpleBlock(32, 32, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=True)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator8(nn.Module):
    def __init__(self):
        super(Generator8, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(512,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator16(nn.Module):
    def __init__(self):
        super(Generator16, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(512,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator32(nn.Module):
    def __init__(self):
        super(Generator32, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(256,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator64(nn.Module):
    def __init__(self):
        super(Generator64, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(128,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator128(nn.Module):
    def __init__(self):
        super(Generator128, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(64,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator256(nn.Module):
    def __init__(self, modify_sequence=None, feature_layers=None):
        super(Generator256, self).__init__()

        if modify_sequence is None:
            def modify_sequence(name, x): return x

        truncated_layers = [
            ('0', NormConvBlock(512, 512, kernel_size=4, padding=3)),
            ('1', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('2', NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1)),
            ('3', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('4', NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1)),
            ('5', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('6', NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1)),
            ('7', NormConvBlock(256, 256, kernel_size=3, padding=1)),
            ('8', NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1)),
            ('9', NormConvBlock(128, 128, kernel_size=3, padding=1)),
            ('10', NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1)),
            ('11', NormConvBlock(64, 64, kernel_size=3, padding=1)),
            ('12', NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1)),
            ('13', NormConvBlock(32, 32, kernel_size=3, padding=1))
        ][:feature_layers]

        self.features = nn.Sequential(OrderedDict(modify_sequence('features',
                                                                  truncated_layers)))

        self.output = nn.Sequential(OrderedDict(modify_sequence('output', [
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(truncated_layers[-1][1].conv.out_channels,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3)),
            ('hardtanh', nn.Hardtanh())
        ])))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Generator1024(nn.Module):
    def __init__(self):
        super(Generator1024, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
            NormConvBlock(32, 32, kernel_size=3, padding=1),
            NormUpscaleConvBlock(32, 16, kernel_size=3, padding=1),
            NormConvBlock(16, 16, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
            ('norm', PixelNormLayer()),
            ('conv', nn.Conv2d(16,
                               3,
                               kernel_size=1,
                               padding=0,
                               bias=False)),
            ('wscale', WScaleLayer(3))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x
