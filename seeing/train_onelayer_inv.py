import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
from torch.nn.functional import mse_loss, cosine_similarity
from collections import defaultdict
from . import encoder_net, setting
from . import nethook, zdataset, pbar
from . import proggan, customnet, parallelfolder
from .encoder_loss import cor_square_error
from torchvision import transforms, models
from .pidfile import exit_if_job_done, mark_job_done

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
parser.add_argument('--invert_layer', type=int, help='Layer to invert',
        default=1)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
args = parser.parse_args()

global_seed = 1
invert_layer = args.invert_layer
expname = 'invert_layer_%d_cse' % invert_layer
expdir = os.path.join('results', args.model, expname)
os.makedirs(expdir, exist_ok=True)

lr_milestones = [20, 60]  # Reduce learning rate after 20 and 60 epochs
if invert_layer == 15 or invert_layer == 2:
    lr_milestones = [60, 80]

def main():
    torch.manual_seed(global_seed)
    pbar.print('Training %s' % expdir)

    # Load a progressive GAN
    generator = setting.load_proggan(args.model)
    # Make a subset model with only some layers.
    if invert_layer == 1:
        s_maker = IdentityLayer()
    else:
        s_maker = nethook.subsequence(generator,
                last_layer='layer%d' % (invert_layer - 1))
    r_maker = nethook.subsequence(generator,
            first_layer='layer%d' % invert_layer,
            last_layer='layer%d' % invert_layer)
    r_decoder = nethook.subsequence(generator,
            first_layer='layer%d' % (invert_layer + 1))

    # Make an encoder model.
    if invert_layer == 1:
        encoder = encoder_net.Layer1toZNormEncoder()
    else:
        channels = [512, # "layer0" is z
            512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 3]
        encoder = encoder_net.LayerNormEncoder(
            channels[invert_layer],
            channels[invert_layer - 1],
            stride=(2 if (invert_layer % 2 == 1 and invert_layer < 15) else 1),
            skip_conv3=(invert_layer == 2),
            skip_pnorm=(invert_layer == 15))

    # Move models to GPU
    for m in [generator, encoder, s_maker, r_maker, r_decoder]:
        m.cuda()

    # Set up a training data loader: unending batches of random z.
    batch_size = 32
    train_loader = training_loader(generator, batch_size)

    # Test data loader is finite, fixed set of z.
    test_loader = testing_loader(generator, batch_size)

    # Set up optimizer
    set_requires_grad(False, generator, s_maker, r_maker, r_decoder)
    learning_rate = args.lr
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1)

    epoch_batches = 100
    num_epochs = 100
    # img_elems = 256*256*3
    # rep_elems = 8*8*512
    # alpha = float(rep_elems) / (rep_elems + img_elems)
    for epoch, epoch_loader in enumerate(pbar(
            epoch_grouper(train_loader, epoch_batches, num_epochs=1+num_epochs),
            total=(1+num_epochs))):
        # Training loop (for 0th epoch, do no training, just testing)
        if epoch > 0:
            for (z_batch,) in pbar(epoch_loader, total=epoch_batches):
                (z_batch,) = [d.cuda() for d in [z_batch]]
                loss = encoder_loss(z_batch, s_maker, r_maker, encoder)
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
                        encoder_loss(z_batch, s_maker, r_maker, encoder).item())
                for name, mloss in monitor_losses(
                        z_batch, encoder, s_maker, r_maker, r_decoder).items():
                    losses[name] += nb * mloss.item()
                if epoch % 10 == 0 and i == 0:
                    visualize_results(epoch, z_batch,
                            encoder, s_maker, r_maker, r_decoder)
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

def visualize_results(epoch, z_batch, encoder, s_maker, r_maker, r_decoder):
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    true_s = s_maker(z_batch)
    true_r = r_maker(true_s)
    image_batch = r_decoder(true_r)
    estimated_s = encoder(true_r)
    estimated_r = r_maker(estimated_s)
    estimated_image = r_decoder(estimated_r)
    # For 6 images of the batch, save four images, plus an .npz file.
    for i in range(min(len(z_batch), 6)):
        for name, im in [
                ('epoch_%d_%d_g.png', image_batch),
                ('epoch_%d_%d_r.png', estimated_image),
                ]:
            save_tensor_image(im[i], os.path.join(dirname, name % (epoch, i)))
        numpy.savez(os.path.join(dirname, 'epoch_%d_%d.npz' % (epoch, i)),
            true_z=z_batch[i].cpu().numpy(),
            true_s=true_s[i].cpu().numpy(),
            true_r=true_r[i].cpu().numpy(),
            estimated_s=estimated_s[i].cpu().numpy(),
            estimated_r=estimated_r[i].cpu().numpy())
    shutil.copy(os.path.join(os.path.dirname(__file__), 'lightbox.html'),
            os.path.join(dirname, '+lightbox.html'))

def save_tensor_image(img, filename):
    np_data = ((img.permute(1, 2, 0) / 2 + 0.5) * 255).byte().cpu().numpy()
    PIL.Image.fromarray(np_data).save(filename)

def encoder_loss(z_batch, s_maker, r_maker, encoder):
    true_s = s_maker(z_batch)
    true_r = r_maker(true_s)
    recovered_s = encoder(true_r)
    recovered_r = r_maker(recovered_s)
    return (cor_square_error(true_s, recovered_s)
            + 0.01 * cor_square_error(true_r, recovered_r))

def monitor_losses(z_batch, encoder, s_maker, r_maker, r_decoder):
    true_s = s_maker(z_batch)
    true_r = r_maker(true_s)
    true_image = r_decoder(true_r)
    recovered_s = encoder(true_r)
    recovered_r = r_maker(recovered_s)
    recovered_image = r_decoder(recovered_r)
    return dict(
            loss_cs=cor_square_error(true_s, recovered_s),
            loss_cr=cor_square_error(true_r, recovered_r),
            loss_p=mse_loss(true_image, recovered_image))

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

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return

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
