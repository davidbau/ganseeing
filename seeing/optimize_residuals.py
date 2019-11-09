import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
from collections import OrderedDict
from numbers import Number
from torch.nn.functional import mse_loss, l1_loss
from . import pbar
from . import zdataset
from . import proggan, customnet, parallelfolder
from . import encoder_net, encoder_loss, setting
from torchvision import transforms, models
from torchvision.models.vgg import model_urls
from .pidfile import exit_if_job_done, mark_job_done
from . import nethook
from .pidfile import exit_if_job_done, mark_job_done
from .encoder_loss import cor_square_error
from .nethook import InstrumentedModel

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
parser.add_argument('--image_number', type=int, help='Image number',
        default=95)
parser.add_argument('--image_source', #choices=['val', 'train', 'gan', 'test'],
        default='test')
parser.add_argument('--redo', type=int, help='Nonzero to delete done.txt',
        default=0)
parser.add_argument('--residuals', nargs='*', help='Residuals to adjust',
        default=None)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
parser.add_argument('--halfsize', type=int,
        help='Set to 1 for half size enoder',
        default=0)
parser.add_argument('--snapshot_every', type=int,
        help='only generate snapshots every n iterations',
        default=1000)
args = parser.parse_args()

num_steps = 3000
lr_milestones = [800, 1200, 1800]
residuals = (args.residuals if args.residuals is not None
        else ['layer1', 'layer2', 'layer3'])
global_seed = 1
learning_rate = args.lr
image_number = args.image_number
expgroup = 'optimize_residuals'
# Use an explicit directory name for a different selection of residuals.
if args.residuals is not None:
    expgroup += '_' + '_'.join(residuals)
imagetypecode = (dict(val='i', train='n', gan='z', test='t')
        .get(args.image_source, args.image_source[0]))
expname = 'opt_%s_%d' % (imagetypecode, image_number)
expdir = os.path.join('results', args.model, expgroup, 'cases', expname)
sumdir = os.path.join('results', args.model, expgroup,
        'summary_%s' % imagetypecode)
os.makedirs(expdir, exist_ok=True)
os.makedirs(sumdir, exist_ok=True)

# First load single image optimize (load via test ParallelFolder dataset).

def main():
    pbar.print('Running %s' % expdir)
    delete_log()

    # Grab a target image
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    loaded_image, loaded_z = setting.load_test_image(image_number,
            args.image_source, model=args.model)
    visualize_results((image_number, 'target'),
            loaded_image[0], summarize=True)

    # Load the pretrained generator model.
    gan_generator = setting.load_proggan(args.model)
    # We will wrap this model
    unwrapped_H = nethook.subsequence(gan_generator, last_layer='layer4')
    # Edit the output of this layer
    F = nethook.subsequence(gan_generator, first_layer='layer5')

    # Load a pretrained gan inverter
    encoder = nethook.InstrumentedModel(
            encoder_net.HybridLayerNormEncoder(halfsize=args.halfsize))
    encoder.load_state_dict(torch.load(os.path.join('results', args.model,
       'invert_hybrid_cse/snapshots/epoch_1000.pth.tar'))['state_dict'])
    encoder.eval()
    E = nethook.subsequence(encoder.model, last_layer='resnet')
    D = nethook.subsequence(encoder.model, first_layer='inv4')

    # Also make a conv features model from pretrained VGG
    vgg = models.vgg16(pretrained=True)
    VF = nethook.subsequence(vgg.features, last_layer='20')

    # Move models and data to GPU
    for m in [F, unwrapped_H, E, D, VF]:
        m.cuda()

    # Some constants for the GPU
    with torch.no_grad():
        # Our true image is constant
        true_p = loaded_image.cuda()
        # Invert our image once!
        init_r = E(true_p)
        init_z = D(init_r)
        # Compute our features once!
        true_v = VF(true_p)
        # For GAN-generated images we have ground truth.
        if loaded_z is None:
            true_z = None
            true_r = None
            true_r1, true_r2, true_r3 = None, None, None
        else:
            true_z = loaded_z.cuda()
            with InstrumentedModel(unwrapped_H) as inst_H:
                inst_H.retain_layers(['layer1', 'layer2', 'layer3'])
                true_r = inst_H(true_z)
                true_r1, true_r2, true_r3 = [inst_H.retained_layer(n)
                        for n in ['layer1', 'layer2', 'layer3']]

    # The model we learn are the top-level parameters of this wrapped model.
    H = encoder_net.ResidualGenerator(
            unwrapped_H, init_z, residuals)
    H.eval()
    H.cuda()

    # Set up optimizer
    set_requires_grad(False, F, H, E, D, VF)
    parameters = OrderedDict(H.named_parameters(recurse=False))
    for n, p in parameters.items():
        p.requires_grad = True
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.5)

    # Phase 1: find a better r4 by seeking d1, d2, d3, etc.
    for step_num in pbar(range(num_steps + 1)):
        current_r = H()
        current_p = F(current_r)
        current_v = VF(current_p)

        loss_p = l1_loss(true_p, current_p)
        loss_v = l1_loss(true_v, current_v)
        loss_z = H.dz.pow(2).mean() if hasattr(H, 'dz') else 0
        loss_1 = H.d1.pow(2).mean() if hasattr(H, 'd1') else 0
        loss_2 = H.d2.pow(2).mean() if hasattr(H, 'd2') else 0
        loss_3 = H.d3.pow(2).mean() if hasattr(H, 'd3') else 0
        loss_4 = H.d4.pow(2).mean() if hasattr(H, 'd4') else 0
        loss_r = mse_loss(init_r, current_r)
        loss = (loss_p + loss_v + loss_z + loss_1 + loss_2 + loss_3 + loss_4)

        all_loss = dict(loss=loss, loss_v=loss_v, loss_p=loss_p,
                loss_r=loss_r,
                loss_z=loss_z,
                loss_1=loss_1,
                loss_2=loss_2,
                loss_3=loss_3,
                loss_4=loss_4
                )
        all_loss = { k: v.item() for k, v in all_loss.items()
                if v is not 0 }

        if (step_num % args.snapshot_every == 0) or (step_num == num_steps):
            with torch.no_grad():
                if true_r is not None:
                    all_loss['err_r'] = cor_square_error(current_r, true_r
                            ) * 100
                all_loss['err_p'] = (current_p - true_p).pow(2).mean() * 100
                log_progress('%d ' % step_num + ' '.join(
                    '%s=%.3f' % (k, all_loss[k])
                    for k in sorted(all_loss.keys())), phase='a')
                visualize_results((image_number, 'a', step_num), current_p,
                        summarize=(step_num in [0, num_steps]))
                checkpoint_dict = OrderedDict(all_loss)
                for s in residuals:
                    s = s.replace('layer', '')
                    checkpoint_dict['init_%s' % s] = getattr(H, 'init_' + s)
                    checkpoint_dict['d_%s' % s] = getattr(H, 'd' + s)
                    checkpoint_dict['current_%s' % s] = (
                            getattr(H, 'init_' + s) + getattr(H, 'd' + s))
            save_checkpoint(
                phase='a',
                step=step_num,
                current_r=current_r,
                current_p=current_p,
                true_z=true_z,
                true_r=true_r,
                true_p=true_p,
                lr=learning_rate,
                optimizer=optimizer.state_dict(),
                **checkpoint_dict)

        optimizer.zero_grad()
        loss.backward()
        if step_num < num_steps:
            optimizer.step()
            scheduler.step()

def delete_log():
    try:
        os.remove(os.path.join(expdir, 'log.txt'))
    except:
        pass

def log_progress(s, phase='a'):
    with open(os.path.join(expdir, 'log.txt'), 'a') as f:
        f.write(phase + ' ' + s + '\n')
    pbar.print(s)

def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'step_%s_%d.pth.tar' % (kwargs['phase'], kwargs['step'])
    torch.save(kwargs, os.path.join(dirname, filename))
    # Also save as .mat file for analysis.
    numeric_data = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if isinstance(v, (Number, numpy.ndarray, torch.Tensor))}
    filename = 'step_%s_%d.npz' % (kwargs['phase'], kwargs['step'])
    numpy.savez(os.path.join(dirname, filename), **numeric_data)

def visualize_results(step, img, summarize=False):
    if isinstance(step, tuple):
        filename = '%s.png' % ('_'.join(str(i) for i in step))
    else:
        filename = '%s.png' % str(step)
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    save_tensor_image(img, os.path.join(dirname, filename))
    lbname = os.path.join(dirname, '+lightbox.html')
    if not os.path.exists(lbname):
        shutil.copy(os.path.join(os.path.dirname(__file__),
            'lightbox.html'), lbname)
    if summarize:
        save_tensor_image(img, os.path.join(sumdir, filename))
        lbname = os.path.join(sumdir, '+lightbox.html')
        if not os.path.exists(lbname):
            shutil.copy(os.path.join(os.path.dirname(__file__),
                'lightbox.html'), lbname)

def save_tensor_image(img, filename):
    if len(img.shape) == 4:
        img = img[0]
    np_data = ((img.permute(1, 2, 0) / 2 + 0.5) * 255
            ).clamp(0, 255).byte().cpu().numpy()
    PIL.Image.fromarray(np_data).save(filename)

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isintance(model, torch.nn.Parameter):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def edit(x):
    x = x.clone()
    x[:,EDIT_UNITS] = 0
    return x

#unit_level99 = {}
#for cls in ablation_units:
#    corpus = numpy.load('reltest/churchoutdoor/layer4/ace/%s/corpus.npz' % cls)


if __name__ == '__main__':
    exit_if_job_done(expdir, redo=args.redo)
    main()
    mark_job_done(expdir)
