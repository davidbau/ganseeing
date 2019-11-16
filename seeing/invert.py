import torch, itertools, os
from collections import defaultdict, OrderedDict
from gan_training import checkpoints
from . import nethook, pbar, LBFGS
from seeing.nethook import InstrumentedModel
from torch.nn.functional import mse_loss

def refine_z_lbfgs(init_z, G, target_x, F=None, lambda_f=0,
        R=None, lambda_r=0, num_steps=1000, quiet=False,
        show=None, show_every=100):
    '''
    Starting at init_z, uss LBFGS to find a z for which G(z) -> target_x.
    By default uses l1_loss, but can also mse(F(G(z)), F(target_x))
    '''
    z = init_z.clone()
    parameters = [z]
    nethook.set_requires_grad(False, G)
    nethook.set_requires_grad(True, *parameters)
    if lambda_f:
        with torch.no_grad():
            target_f = F(target_x)
            nethook.set_requires_grad(False, F)

    optimizer = LBFGS.FullBatchLBFGS(parameters)

    def closure():
        optimizer.zero_grad()
        current_x = G(z)
        loss = torch.nn.functional.l1_loss(target_x, current_x)
        if lambda_f:
            loss += torch.nn.functional.mse_loss(target_f, F(current_x)
                    ) * lambda_f
        if lambda_r:
            loss += R(z) * lambda_r
        return loss

    pb = (lambda x: x) if quiet else pbar

    with torch.enable_grad():
        for step_num in pb(range(num_steps + 1)):
            if step_num == 0:
                loss = closure()
                loss.backward()
                lr, F_eval, G_eval = 0, 0, 0
            else:
                options = {'closure': closure, 'current_loss': loss,
                        'max_ls': 10}
                loss, _, lr, _, _, _, _, _ = optimizer.step(options)
            if show and (step_num % show_every == 0):
                with torch.no_grad():
                    show(x=G(z), z=z, loss=loss, it=step_num)

    return z

def split_gen_layers(enc, gen, layername):
    '''
    Given an inverter layername, splits the generator sequence into three:
    (1) generator sequence before the layers to be inverted
    (2) sequence of generator layers to be inverted by enc[layername]
    (3) generator sequence after the layers to be inverted
    '''
    info = list(enc.inverse_info().items())
    index = [i for i, (n, v) in enumerate(info) if n == layername][0]
    upto_layer = info[index - 1][1]['first_layer'] if index > 0 else None
    args = info[index][1]
    first_layer = args['first_layer']
    layers = nethook.subsequence(gen,
         first_layer=first_layer, upto_layer=upto_layer)
    prev_layers = nethook.subsequence(gen, upto_layer=first_layer)
    next_layers = (nethook.subsequence(gen, first_layer=upto_layer) if
            upto_layer else torch.nn.Sequential())
    return prev_layers, layers, next_layers

def last_gen_layername(enc, gen, layername):
    _, layers, _ = split_gen_layers(enc, gen, layername)
    return [n for n, c in layers.named_children()][-1]

def layers_after(enc, layername):
    layernames = [n for n, c in enc.named_children()]
    index = layernames.index(layername)
    if index + 1 < len(layernames):
        return nethook.subsequence(enc, layernames[index + 1])
    else:
        return torch.nn.Sequential()

def train_inv_layer(enc, gen, dists, layername, combine_z=None,
        batch_size=100, test_batches=10, checkpoint_dir='ckpts',
        resume_from=None, logfile=None, **kwargs):
    '''
    Inverts a single layer of a multilayer inverter.
    Both enc and should be a nn.Sequential subclasses, and
    layername specifies the layer of enc to train.  That layer
    of enc will be trained to invert a set of gen layers.
    Which layers specifically are determined by split_gen_layers,
    which depends on enc.inverse_info(), to specify how each inverter
    layer relates to layers of gen.
    '''
    if logfile is None:
        logfile = os.path.join(checkpoint_dir, 'log.txt')
    prev_layers, layers, next_layers = split_gen_layers(enc, gen, layername)
    inv = getattr(enc, layername)
    device = next(gen.parameters()).device
    args = enc.inverse_info()[layername]
    kwargs_out = {k: v for k, v in args.items() if k != 'first_layer'}
    kwargs_out.update(kwargs)
    if 'x_weight' not in kwargs_out:
        kwargs_out['x_weight'] = 0
    nethook.set_requires_grad(False, layers, prev_layers, next_layers)
    if combine_z is not None:
        nethook.set_requires_grad(False, combine_z)
        prev_layers_old = prev_layers
        prev_layers = lambda *a: prev_layers_old(combine_z(*a))
    zsampler = infinite_sampler(dists, prev_layers, batch_size, device)
    tsample = test_sampler(dists, prev_layers, batch_size, test_batches, device)
    train_inverse(inv, layers, zsampler, test_sampler=tsample,
            resume_from=resume_from,
            checkpoint_dir=checkpoint_dir, logfile=logfile,
            **kwargs_out)

def train_inv_joint(enc, gen, dists, combine_z=None,
        inv_layer=None, gen_layer=None,
        batch_size=50, test_batches=10,
        checkpoint_dir='ckpts',
        logfile=None, **kwargs):
    if logfile is None:
        logfile = os.path.join(checkpoint_dir, 'log.txt')
    device = next(gen.parameters()).device
    zsampler = infinite_sampler(dists, combine_z, batch_size, device)
    tsample = test_sampler(dists, combine_z, batch_size, test_batches, device)

    with InstrumentedModel(gen) as G, InstrumentedModel(enc) as E:
        G.retain_layer(gen_layer, detach=False)
        nethook.set_requires_grad(False, G)
        E.retain_layer(inv_layer, detach=False)
        train_inverse(E, G, zsampler, inv_layer, gen_layer,
            r_weight=1.0, ir_weight=1.0, test_sampler=tsample,
            checkpoint_dir=checkpoint_dir,
            checkpoint_selector=lambda x: x.model,
            logfile=logfile, **kwargs)

def train_inverse(inv, gen, sampler, inv_layer=None, gen_layer=None,
        z_weight=1.0, x_weight=1.0, r_weight=0.0, ir_weight=0.0, reg_weight=0.0,
        adjust_z=None, regularize_z=None,
        test_sampler=None, lr=0.01, lr_milestones=None,
        epoch_batches=100, num_epochs=100, save_every=50,
        logfile=None,
        checkpoint_dir=None, checkpoint_selector=None, resume_from=None):
    '''
    To set this up:
    inv and gen should both be instrumented models,
    and inv layer and gen layer should be retained on both of them
    without detach.
    '''
    if lr_milestones is None:
        lr_milestones = []
    optimizer = torch.optim.Adam(inv.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1)

    ckpts = checkpoints.CheckpointIO(checkpoint_dir)
    ckpts.register_modules(optimizer=optimizer, scheduler=scheduler,
            inv=inv if not checkpoint_selector else checkpoint_selector(inv))
    if resume_from is not None:
        ckpts.load('ckpt_%d.pt' % resume_from)
        start_epoch = resume_from + 1
    else:
        start_epoch = 0

    def inv_loss(z):
        return sum(loss * weight
                for loss, weight in component_loss(z).values())

    def component_loss(true_z):
        observed_x = gen(true_z)
        if gen_layer:
            true_r = gen.retained_layer(gen_layer, clear=True)
        estimated_z = inv(observed_x)
        if inv_layer:
            inverted_r = inv.retained_layer(inv_layer, clear=True)
        if adjust_z:
            fixed_z = adjust_z(estimated_z, true_z)
        else:
            fixed_z = estimated_z
        if x_weight or r_weight:
            reconstructed_x = gen(fixed_z)
        if gen_layer:
            reconstructed_r = gen.retained_layer(gen_layer, clear=True)
        losses = OrderedDict()
        if reg_weight:
            losses['reg'] = (regularize_z(estimated_z, true_z), reg_weight)
        if z_weight:
            losses['z'] = (mse_loss(true_z, estimated_z), z_weight)
        if ir_weight:
            losses['ir'] = (cor_square_error(true_r, inverted_r), ir_weight)
        if x_weight:
            losses['x'] = (mse_loss(observed_x, reconstructed_x), x_weight)
        if r_weight:
            losses['r'] = (cor_square_error(true_r, reconstructed_r), r_weight)
        return losses

    with torch.no_grad():
        for epoch, epoch_loader in pbar(
                epoch_grouper(sampler, epoch_batches, num_epochs=1+num_epochs,
                    start_epoch=start_epoch),
                total=(1+num_epochs-start_epoch)):
            if epoch > 0:
                for (z_batch,) in epoch_loader:
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        loss = inv_loss(z_batch)
                        loss.backward()
                        pbar.post(l=loss.item())
                        optimizer.step()
                scheduler.step()
            if test_sampler is not None:
                stats = MeanStats()
                for (z_batch,) in test_sampler:
                    stats.add(component_loss(z_batch), len(z_batch))
                logline = stats.logline(epoch)
                pbar.print(logline)
                if logfile is not None:
                    with open(logfile, 'a') as f:
                        f.write(logline + '\n')
            elif epoch > 0:
                pbar.print('%d: loss=%4g' % (epoch, loss.item()))
            if epoch % save_every == 0 or epoch == num_epochs:
                ckpts.save(epoch, 'ckpt_%d.pt' % epoch)

def infinite_sampler(dists, f, batch_size, device):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    while True:
        zs = [dist.sample([batch_size]).to(device) for dist in dists]
        r = zs[0] if f is None else f(*zs)
        yield (r,)

def test_sampler(dists, f, batch_size, test_batches, device):
    class TestSampler():
        def __init__(self):
            self.num_batches = test_batches
            self.zs_batches = [
                    dist.sample([test_batches, batch_size]).to(device)
                    for dist in dists]
        
        def __iter__(self):
            for i in range(self.num_batches):
                zs = [uncombined[i] for uncombined in self.zs_batches]
                r = zs[0] if f is None else f(*zs)
                yield (r,)
    return TestSampler()

def epoch_grouper(loader, epoch_size, num_epochs=None, start_epoch=0):
    '''
    To use with an infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = start_epoch
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield epoch, itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return

def cor_square_error(x, y, eps=1e-12):
    # Analogous to MSE, but in terms of Pearson's correlation
    return (1.0 - torch.nn.functional.cosine_similarity(x, y, eps=eps)).mean()


class MeanStats:
    def __init__(self):
        self.tally = defaultdict(float)
        self.count = 0

    def add(self, c, size):
        for n, (loss, weight) in c.items():
            self.tally[n] += loss.item() * size
        self.count += size

    def summary(self):
        return {n: v / self.count for n, v in self.tally.items()}

    def logline(self, i=None):
        prefix = '' if i is None else '%d: ' % i
        return prefix + ' '.join('%s=%4g' % (n, v)
                for n, v in self.summary().items())
