import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
from torch.nn.functional import mse_loss
from collections import defaultdict, OrderedDict
from . import encoder_net, setting
from .encoder_loss import cor_square_error
from . import zdataset, pbar, nethook
from . import proggan, customnet, parallelfolder
from torchvision import transforms, models
from .pidfile import exit_if_job_done, mark_job_done

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
parser.add_argument('--estimate', nargs='+', default=None,
        help='estimation losses: subset of z, layer1, layer2, layer3, layer4')
parser.add_argument('--recover', nargs='+', default=None,
        help='recovey losses: subset of layer1, layer2, layer3, layer4, x')
args = parser.parse_args()

global_seed = 1
variant = None
if args.estimate == args.recover == None:
    variant = ['cse']
    args.estimate = ['layer4']
    args.recover = ['z', 'layer4', 'x']
if args.estimate is None:
    args.estimate = []
if args.recover is None:
    args.recover = []
if variant is None:
    variant = sum([
        [prefix + k.replace('layer', '') for k in lst]
        for prefix, lst in [
            ('e', args.estimate),
            ('r', args.recover)]
        ], [])
# make a directory name like 'invert_hybrid_s1_rz_r2_r4_rx'
# default '_cse' corresponds to '_i4_rz_r4_rx'
expname = 'invert_hybrid_' + '_'.join(variant)
expdir = os.path.join('results', args.model, expname)
os.makedirs(expdir, exist_ok=True)

num_epochs = 1000
lr_milestones = [50, 150, 350]  # Reduce learning rate after these epochs

def main():
    torch.manual_seed(global_seed)
    pbar.print('Training %s' % expname)

    # Load a progressive GAN
    generator = setting.load_proggan(args.model)
    # Make a stacked encoder, and load each of its pretrained layers
    encoder = encoder_net.HybridLayerNormEncoder()
    for layernum in range(1, 4+1):
        layer = getattr(encoder, 'inv%d' % layernum)
        layer_filename = (os.path.join('results', args.model,
                'invert_layer_%d_cse/snapshots/epoch_100.pth.tar' % layernum))
        pbar.print('Loading %s' % layer_filename)
        layer.load_state_dict(torch.load(layer_filename)['state_dict'])
    resnet_filename = (os.path.join('results', args.model,
            'invert_over5_resnet/snapshots/epoch_100.pth.tar'))
    pbar.print('Loading %s' % resnet_filename)
    encoder.resnet.load_state_dict(torch.load(resnet_filename)['state_dict'])

    # Instrument both the encoder and generator model so that we can add
    # extra loss terms based on reconstruction of intermediate layers.
    generator = nethook.InstrumentedModel(generator)
    generator_layer_list = ['layer1', 'layer2', 'layer3', 'layer4']
    for n in args.recover:
        if n not in generator_layer_list and n not in ['x', 'z']:
            generator_layer_list.append(n)
    generator.retain_layers(generator_layer_list, detach=False)
    encoder = nethook.InstrumentedModel(encoder)
    encoder.retain_layers([
            ('resnet', 'layer4'),
            ('inv4', 'layer3'),
            ('inv3', 'layer2'),
            ('inv2', 'layer1')], detach=False)

    # Move models to GPU
    for m in [generator, encoder]:
        m.cuda()

    # Set up a training data loader: unending batches of random z.
    batch_size = 32
    train_loader = training_loader(generator, batch_size)

    # Test data loader is finite, fixed set of z.
    test_loader = testing_loader(generator, batch_size)

    # Set up optimizer
    set_requires_grad(False, generator)
    optimize_conv3 = False
    if optimize_conv3:
        target_params = encoder.parameters()
    else:
        target_params = []
        # The conv3 filters of each layer are redundant with
        # the conv1 in the following layer, so freeze the conv3 layers.
        for n, p in encoder.named_parameters():
            if (n.startswith('inv') and not n.startswith('inv1.')
                    and 'conv3' in n):
                p.requires_grad = False
            else:
                target_params.append(p)
    learning_rate = args.lr
    optimizer = torch.optim.Adam(target_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.5)

    epoch_batches = 100
    for epoch, epoch_loader in enumerate(pbar(
            epoch_grouper(train_loader, epoch_batches), total=(1+num_epochs))):
        # Training loop (for 0th epoch, do no training, just testing)
        if epoch > 0:
            for (z_batch,) in pbar(epoch_loader, total=epoch_batches):
                (z_batch,) = [d.cuda() for d in [z_batch]]
                loss = encoder_loss(z_batch, generator, encoder)
                loss.backward()
                pbar.post(l=loss.item())
                optimizer.step()
            scheduler.step()
        # Testing loop
        with torch.no_grad():
            losses = defaultdict(float)
            count = 0
            for i, (z_batch,) in enumerate(pbar(test_loader)):
                (z_batch,) = [d.cuda() for d in [z_batch]]
                nb = len(z_batch)
                # Some other debugging losses
                count += nb
                losses['loss'] += nb * (
                        encoder_loss(z_batch, generator, encoder).item())
                for name, mloss in monitor_losses(
                        z_batch, generator, encoder).items():
                    losses[name] += nb * mloss.item()
                if epoch % 10 == 0 and i == 0:
                    visualize_results(epoch, z_batch, generator, encoder)
            losses = { name: loss / count for name, loss in losses.items() }
            logline = '%d ' % epoch + ' '.join("%s=%4g" % (name, losses[name])
                    for name in sorted(losses.keys()))
            pbar.print(logline)
            with open(os.path.join(expdir, 'log.txt'), 'a') as f:
                f.write(logline + '\n')
        if epoch % 10 == 0:
            save_checkpoint(
                    epoch=epoch,
                    state_dict=encoder.state_dict(),
                    lr=learning_rate,
                    optimizer=optimizer.state_dict(),
                    **losses)
        if epoch == num_epochs:
            break

def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'epoch_%d.pth.tar' % kwargs['epoch']
    torch.save(kwargs, os.path.join(dirname, filename))

def visualize_results(epoch, true_z, generator, encoder):
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    true_r, estimated_r, recovered_r = (
            generate_and_recover_features(true_z, generator, encoder))
    num_images = 6
    for i in range(min(len(true_z), num_images)):
        for name, im in [
                ('epoch_%d_%d_g.png', true_r['x']),
                ('epoch_%d_%d_r.png', recovered_r['x']),
                ]:
            save_tensor_image(im[i], os.path.join(dirname, name % (epoch, i)))
        rawdat = OrderedDict(sum([
            [(template % k.replace('layer', ''), v[i].cpu().numpy())
                for k, v in feats.items()]
            for template, feats in [
                ('t_%s', true_r),
                ('e_%s', estimated_r),
                ('r_%s', recovered_r)]], []))
        numpy.savez(os.path.join(dirname, 'epoch_%d_%d.npz' % (epoch, i)),
                **rawdat)
    shutil.copy(os.path.join(os.dirname(__file__), 'lightbox.html'),
            os.path.join(dirname, '+lightbox.html'))

def save_tensor_image(img, filename):
    np_data = ((img.permute(1, 2, 0) / 2 + 0.5) * 255).byte().cpu().numpy()
    PIL.Image.fromarray(np_data).save(filename)
    
def generate_recovered(generator, estimated_r):
    return generator(estimated_r['z'])

def generate_and_recover_features(true_z, generator, encoder):
    global args
    true_x = generator(true_z)
    true_r = generator.retained_features(clear=True)
    true_r['z'] = true_z
    true_r['x'] = true_x
    recovered_z = encoder(true_x)
    estimated_r = encoder.retained_features(clear=True)
    estimated_r['z'] = recovered_z
    recovered_x = generate_recovered(generator, estimated_r)
    recovered_r = generator.retained_features(clear=True)
    recovered_r['z'] = recovered_z
    recovered_r['x'] = recovered_x
    return true_r, estimated_r, recovered_r

def monitor_losses(true_z, generator, encoder, all_losses=True):
    global args
    true_r, estimated_r, recovered_r = (
            generate_and_recover_features(true_z, generator, encoder))
    losses = {}
    for layer in estimated_r.keys() if all_losses else args.estimate:
        losses['i' + layer.replace('layer', '')] = (
                cor_square_error(true_r[layer], estimated_r[layer]))
    for layer in recovered_r.keys() if all_losses else args.recover:
        if layer == 'x':
            losses['rx'] = (
                    mse_loss(true_r[layer], recovered_r[layer]))
        else:
            losses['r' + layer.replace('layer', '')] = (
                    cor_square_error(true_r[layer], recovered_r[layer]))
    return losses

def encoder_loss(true_z, generator, encoder):
    return sum(monitor_losses(true_z, generator, encoder,
        all_losses=False).values())

def old_encoder_loss(true_z, generator, encoder):
    true_p = generator(true_z)
    true_r = generator.retained_features(clear=True)['layer4']
    recovered_z = encoder(true_p)
    estimated_r = encoder.retained_features(clear=True)['layer4']
    recovered_p = generator(recovered_z)
    recovered_r = generator.retained_features(clear=True)['layer4']
    loss_z = cor_square_error(true_z, recovered_z)
    loss_ir = cor_square_error(true_r, estimated_r)
    loss_r = cor_square_error(true_r, recovered_r)
    loss_p = mse_loss(true_p, recovered_p)
    return loss_z + loss_ir + loss_r + loss_p

def old_monitor_losses(true_z, generator, encoder):
    true_p = generator(true_z)
    true_r = generator.retained_features(clear=True)['layer4']
    recovered_z = encoder(true_p)
    estimated_r = encoder.retained_features(clear=True)['layer4']
    recovered_p = generator(recovered_z)
    recovered_r = generator.retained_features(clear=True)['layer4']
    return dict(
            loss_z=cor_square_error(true_z, recovered_z),
            loss_ir=cor_square_error(true_r, estimated_r),
            loss_r=cor_square_error(true_r, recovered_r),
            loss_p=mse_loss(true_p, recovered_p))

def training_loader(z_generator, batch_size):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = zdataset.z_dataset_for_model(
                z_generator, size=10000, seed=g_epoch + global_seed)
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=10,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def testing_loader(z_generator, batch_size):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    z_data = zdataset.z_dataset_for_model(
        z_generator, size=1000, seed=global_seed)
    dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=10,
                pin_memory=True)
    return dataloader

def epoch_grouper(loader, epoch_size):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)

def set_requires_grad(requires_grad, *models):
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad

class IdentityLayer(torch.nn.Module):
    def forward(self, x):
        return x

if __name__ == '__main__':
    exit_if_job_done(expdir)
    main()
    mark_job_done(expdir)
