import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
from collections import OrderedDict
from numbers import Number
from torch.nn.functional import mse_loss, l1_loss
from seeing import pbar
from seeing import zdataset, seededsampler
from seeing import proggan, customnet, parallelfolder
from seeing import encoder_net, encoder_loss, setting
from torchvision import transforms, models
from torchvision.models.vgg import model_urls
from seeing.pidfile import exit_if_job_done, mark_job_done
from seeing import nethook, LBFGS
from seeing.encoder_loss import cor_square_error
from seeing.nethook import InstrumentedModel

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--image_number', type=int, help='Image number',
        default=95)
parser.add_argument('--image_source', #choices=['val', 'train', 'gan', 'test'],
        default='test')
parser.add_argument('--redo', type=int, help='Nonzero to delete done.txt',
        default=0)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
parser.add_argument('--halfsize', type=int,
        help='Set to 1 for half size enoder',
        default=0)
parser.add_argument('--lambda_f', type=float, help='Feature regularizer',
        default=0.25)
parser.add_argument('--num_steps', type=int,
        help='run for n steps',
        default=3000)
parser.add_argument('--snapshot_every', type=int,
        help='only generate snapshots every n iterations',
        default=1000)
args = parser.parse_args()


num_steps = args.num_steps
global_seed = 1
image_number = args.image_number
expgroup = 'optimize_lbfgs'
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
    loaded_x, loaded_z = setting.load_test_image(image_number,
            args.image_source, model=args.model)
    visualize_results((image_number, 'target'),
            loaded_x[0], summarize=True)

    # Load the pretrained generator model.
    G = setting.load_proggan(args.model)

    # Load a pretrained gan inverter
    E = nethook.InstrumentedModel(
            encoder_net.HybridLayerNormEncoder(halfsize=args.halfsize))
    E.load_state_dict(torch.load(os.path.join('results', args.model,
       'invert_hybrid_cse/snapshots/epoch_1000.pth.tar'))['state_dict'])
    E.eval()

    G.cuda()
    E.cuda()
    F = E

    torch.set_grad_enabled(False)
    # Some constants for the GPU
    # Our true image is constant
    true_x = loaded_x.cuda()
    # Invert our image once!
    init_z = E(true_x)
    # For GAN-generated images we have ground truth.
    if loaded_z is None:
        true_z = None
    else:
        true_z = loaded_z.cuda()

    current_z = init_z.clone()
    target_x = loaded_x.clone().cuda()
    target_f = F(loaded_x.cuda())
    parameters = [current_z]
    show_every = args.snapshot_every

    nethook.set_requires_grad(False, G, E)
    nethook.set_requires_grad(True, *parameters)
    optimizer = LBFGS.FullBatchLBFGS(parameters)

    def compute_all_loss():
        current_x = G(current_z)
        all_loss = {}
        all_loss['x'] = l1_loss(target_x, current_x)
        all_loss['z'] = 0.0 if not args.lambda_f else (
            mse_loss(target_f, F(current_x)) * args.lambda_f)
        return current_x, all_loss

    def closure():
        optimizer.zero_grad()
        _, all_loss = compute_all_loss()
        return sum(all_loss.values())

    with torch.enable_grad():
        for step_num in pbar(range(num_steps + 1)):
            if step_num == 0:
                loss = closure()
                loss.backward()
            else:
                options = {'closure': closure, 'current_loss': loss,
                        'max_ls': 10}
                loss, _, lr, _, _, _, _, _ = optimizer.step(options)
            if step_num % show_every == 0:
                with torch.no_grad():
                    current_x, all_loss = compute_all_loss()
                    log_progress('%d ' % step_num + ' '.join(
                        '%s=%.3f' % (k, all_loss[k])
                        for k in sorted(all_loss.keys())), phase='a')
                    visualize_results((image_number, 'a', step_num), current_x,
                        summarize=(step_num in [0, num_steps]))
                checkpoint_dict = OrderedDict(all_loss)
                checkpoint_dict['init_z'] = init_z
                checkpoint_dict['target_x'] = target_x
                checkpoint_dict['current_z'] = target_x
                save_checkpoint(
                    phase='a',
                    step=step_num,
                    optimizer=optimizer.state_dict(),
                    **checkpoint_dict)

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
    # TODO: add editing etc.
    if isinstance(step, tuple):
        filename = '%s.png' % ('_'.join(str(i) for i in step))
    else:
        filename = '%s.png' % str(step)
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    save_tensor_image(img, os.path.join(dirname, filename))
    lbname = os.path.join(dirname, '+lightbox.html')
    if not os.path.exists(lbname):
        shutil.copy('seeing/lightbox.html', lbname)
    if summarize:
        save_tensor_image(img, os.path.join(sumdir, filename))
        lbname = os.path.join(sumdir, '+lightbox.html')
        if not os.path.exists(lbname):
            shutil.copy('seeing/lightbox.html', lbname)

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

if __name__ == '__main__':
    exit_if_job_done(expdir, redo=args.redo)
    main()
    mark_job_done(expdir)
