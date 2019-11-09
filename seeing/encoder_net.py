import numpy, math
from . import customnet, nethook
from collections import OrderedDict
import torch.nn
import re

def make_over5_resnet(halfsize=False):
    # A resnet with the global pooling layer replaced by a single 1x1
    # conv layer, to produce a 512x8x8 featuremap.  Also adds a leaky
    # ReLU to better resemble the distribution of r produced by the GAN.
    resnet_depth = 18
    # Make an encoder model.
    def change_out(layers):
        numch = 512 if resnet_depth < 50 else 2048
        ind = [i for i, (n, l) in enumerate(layers) if n == 'layer4'][0] + 1
        layers[ind:] = [('layer5',
            torch.nn.Sequential(OrderedDict([
                ('conv5', torch.nn.Conv2d(numch, 512, kernel_size=1)),
                ('relu5', torch.nn.LeakyReLU(
                    inplace=True, negative_slope=0.2))
            ])))]
        return layers
    encoder = customnet.CustomResNet(
            resnet_depth, modify_sequence=change_out, halfsize=halfsize)
    return encoder

class HybridLayerNormEncoder(torch.nn.Sequential):
    def __init__(self, halfsize=False):
        sequence = [
            ('resnet', make_over5_resnet(halfsize=halfsize)),
            ('inv4', LayerNormEncoder(512, 512)),
            ('inv3', LayerNormEncoder(512, 512, stride=2)),
            ('inv2', LayerNormEncoder(512, 512, skip_conv3=True)),
            ('inv1', Layer1toZNormEncoder())
        ]
        super().__init__(OrderedDict(sequence))

class LayerNormEncoder(torch.nn.Sequential):
    def __init__(self, chan_in, chan_out=None, stride=1,
            skip_conv3=False, skip_pnorm=False):
        if chan_out is None:
            chan_out = chan_in
        sequence = []
        if not skip_pnorm:
            sequence.append(('pnorm', PixelNormLayer()))
        sequence.extend([
            ('conv1', torch.nn.Conv2d(chan_in, chan_out,
                kernel_size=3, padding=1)),
            ('bn1',   torch.nn.BatchNorm2d(chan_out)),
            ('relu1', torch.nn.LeakyReLU(inplace=True, negative_slope=0.2)),
            ('conv2', torch.nn.Conv2d(chan_out, chan_out,
                kernel_size=3, padding=1)),
            ('bn2',   torch.nn.BatchNorm2d(chan_out)),
            ('relu2', torch.nn.LeakyReLU(inplace=True, negative_slope=0.2)),
            ])
        if not skip_conv3:
            sequence.append(
                ('conv3', torch.nn.Conv2d(chan_out, chan_out,
                    kernel_size=1, padding=0, stride=stride)))
        super().__init__(OrderedDict(sequence))
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n.endswith('.bias'):
                    p.zero_()
                elif not n.startswith('bn'):
                    torch.nn.init.kaiming_normal_(p)

class Layer1toZNormEncoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(OrderedDict([
            ('pnorm', PixelNormLayer()),
            ('conv1', torch.nn.Conv2d(512, 512, kernel_size=4, padding=0)),
            ('bn1',   torch.nn.BatchNorm2d(512)),
            ('relu1', torch.nn.LeakyReLU(inplace=True, negative_slope=0.2)),
            ('conv2', torch.nn.Conv2d(512, 512, kernel_size=1, padding=0)),
            ('pnormout', PixelNormLayer())
            ]))
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n.endswith('.bias'):
                    p.zero_()
                elif not n.startswith('bn'):
                    torch.nn.init.kaiming_normal_(p)

class ResidualGenerator(nethook.InstrumentedModel):
    '''
    '''
    def __init__(self, generator, z, residual_layers):
        '''
        ResidualGenerator(generator, z, ['z', 'layer1', 'layer2'])
        Returns a model that computes generator(z), but which has
        additional internal parameters dz, d1, d2, etc, that
        adjust the computation so that the output of layerN is
        adjusted by dN, for example, if a network normally computes

        x = layer4(layer3(layer2(layer1(z)))), then specifying

        the innermost three layers will cause this to compute:

        x = layer4(d3 + layer3(d2 + layer2(d1 + layer1(dz + z))))
        '''
        # First temporarily hook the layers of the generator to
        # collect initial values (and output shapes) of each layer.
        with torch.no_grad(), nethook.InstrumentedModel(generator) as g:
            g.retain_layers([n for n in residual_layers if n != 'z'])
            init_out = g(z)
            init_layers = g.retained_features()
            init_layers['z'] = z
        # Then, permanently hook the layers of the generator to add
        # residual adjustments dz, d1, d2, etc at each layer.
        super().__init__(generator)
        for k, v in init_layers.items():
            # layer3.conv1 -> 3_conv1, shortened name.
            name = k.replace('layer', '', 1).replace('.', '_')
            # 3_conv1 -> self.init_3_conv1, buffer with unperturbed value
            self.register_buffer('init_%s' % name, v.clone())
            # Add parameter 'dz', etc for any variable listed in residuals
            if k in residual_layers:
                # 3_conv1 -> self.d3_conv1, parameter initialized to 0
                dname = 'd' + name
                setattr(self, dname, torch.nn.Parameter(torch.zeros_like(v)))
                # Change model to add self.d[name] after computing layer k.
                if k != 'z':
                    self.edit_layer(k, add_adjustment, attr=dname)

    def forward(self):
        return super().forward(self.init_z + getattr(self, 'dz', 0))

class FixedGANPriorGenerator(nethook.InstrumentedModel):
    '''
    Combines the ideas of ResidualGenerator and GANPriorRUNetGenerator.
    '''
    def __init__(self, generator, z, additive=False):
        self.additive = additive
        # To begin with, we want to glue skip connections into our
        # generator.  Modify its 'forward' method to accept skip args.
        generator = SkipAdjustedSequence(generator)
        skip_layers = ['layer8', 'layer10', 'layer12', 'layer14']
        # Gather some initial values programmatically with a temporary hook.
        with torch.no_grad(), nethook.InstrumentedModel(generator) as g:
            g.retain_layers([n for n in (
                skip_layers)
                if n != 'z'])
            init_out = g(z)
            init_layers = g.retained_features()
            init_layers['z'] = z
        # Then, permanently hook the layers of the generator to add
        # residual adjustments dz, d1, d2, etc at each layer.
        super().__init__(generator)
        for k, v in init_layers.items():
            # Record all the init_N values for reporting and reference.
            name = k.replace('layer', '', 1).replace('.', '_')
            self.register_buffer('init_%s' % name, v.clone())
        # Now the deep image prior u-net side, melding with pixels.
        # Start with a fixed random featuremap
        seed = 1
        rng = numpy.random.RandomState(seed)
        self.register_buffer('noise', torch.from_numpy(
            rng.randn(1, 32, 256, 256)).float())
        # put through 8 conv layers
        self.down14 = UnetDownsample(32, 32, self.init_14.size(1), 4, stride=1,
                rng=rng)
        self.down12 = UnetDownsample(32, 32, self.init_12.size(1), 4, rng=rng)
        self.down10 = UnetDownsample(32, 32, self.init_10.size(1), 4, rng=rng)
        self.down8  = UnetDownsample(32, 0, self.init_8.size(1), 4, rng=rng)
        # Finally, also retain the editing layer, layer4
        self.train(True)

    def forward(self, z=None):
        # First run the deep-image-prior noise maker.
        z14, s14 = self.down14(self.noise)
        z12, s12 = self.down12(z14)
        z10, s10 = self.down10(z12)
        _, s8   = self.down8(z10)
        # _, s6   = self.down6(z8)
        # _, s4   = self.down4(z6)
        if z is None:
            z = self.init_z
        # Collect together adjustments before running the generator.
        if self.additive:
            x = super().forward(z,
                    add_layer8=(s8),
                    add_layer10=(s10),
                    add_layer12=(s12),
                    add_layer14=(s14),
                    )
        else:
            x = super().forward(z,
                    mult_layer8=(1 + s8),
                    mult_layer10=(1 + s10),
                    mult_layer12=(1 + s12),
                    mult_layer14=(1 + s14),
                    )
        return dict(
                x=x,
                # Note: this is retained before it is multiplied
                s8=s8,
                s10=s10,
                s12=s12,
                s14=s14)

class BaselineTunedDirectGenerator(nethook.InstrumentedModel):
    '''
    Combines the ideas of ResidualGenerator and GANPriorRUNetGenerator.
    '''
    def __init__(self, generator, z, tune_layers=None):
        # To begin with, we want to glue skip connections into our
        # generator.  Modify its 'forward' method to accept skip args.
        generator = SkipAdjustedSequence(generator)
        if tune_layers is None:
            tune_layers = ['layer8', 'layer10', 'layer12', 'layer14']
        # Gather some initial values programmatically with a temporary hook.
        with torch.no_grad(), nethook.InstrumentedModel(generator) as g:
            g.retain_layers([n for n in (
                tune_layers)
                if n != 'z'])
            init_out = g(z)
            init_layers = g.retained_features()
            init_layers['z'] = z
        # Then, permanently hook the layers of the generator to add
        # residual adjustments dz, d1, d2, etc at each layer.
        super().__init__(generator)
        self.adjustments = []
        for k, v in init_layers.items():
            # Record all the init_N values for reporting and reference.
            name = k.replace('layer', '', 1).replace('.', '_')
            dname = 'd%s' % name
            self.register_buffer('init_%s' % name, v.clone())
            if 'layer' in k:
                self.adjustments.append(('mult_%s' % k, dname))
            setattr(self, dname, torch.nn.Parameter(torch.zeros_like(v)))
        self.train(True)

    def forward(self, z=None, **kwargs):
        # Collect together adjustments before running the generator.
        kwadj = {k: 1 +
            kwargs.get(dname, getattr(self, dname)) # Allow kwargs to override
            for k, dname in self.adjustments}
        if z is None:
            z = self.init_z
        x = super().forward(z, **kwadj)
        kwout = {dname: getattr(self, dname) for k, dname in self.adjustments}
        return dict(x=x, **kwout)

class SkipAdjustedSequence(torch.nn.Sequential):
    def __init__(self, sequential, share_weights=False):
        '''
        Creates a subsequence of a pytorch Sequential model, copying over
        modules together with parameters for the subsequence.  Only
        modules from first_layer to last_layer (inclusive) are included.

        If share_weights is True, then references the original modules
        and their parameters without copying them.  Otherwise, by default,
        makes a separate brand-new copy.
        '''
        included_children = OrderedDict()
        for name, layer in sequential._modules.items():
            included_children[name] = layer if share_weights else (
                        copy.deepcopy(layer))
        if not len(included_children):
            raise ValueError('Empty subsequence')
        super().__init__(OrderedDict(included_children))

    def forward(this, x, **kwargs):
        '''
        Runs the sequence, except after each step 'layer', adds
        any 'add_layer' value from kwargs to the output; similarly
        multiplies any 'mult_layer' value from kwargs if present.
        '''
        seen = set()
        for name, layer in this._modules.items():
            x = layer(x)
            add = kwargs.get('add_' + name, None)
            if add is not None:
                x = x + add
                seen.add('add_' + name)
            mult = kwargs.get('mult_' + name, None)
            if mult is not None:
                x = x * mult
                seen.add('mult_' + name)
        for name in kwargs.keys():
            assert name in seen, '%s not applied' % name
        return x

class PixelNormLayer(torch.nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

def add_adjustment(x, idecoder, attr):
    adjustment = getattr(idecoder, attr)
    x = x + adjustment
    return x

