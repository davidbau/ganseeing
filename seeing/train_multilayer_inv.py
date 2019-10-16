import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
from torch.nn.functional import mse_loss
from . import pbar, setting
from . import encoder_net
from . import nethook, zdataset
from . import proggan, customnet, parallelfolder
from torchvision import transforms, models
from torchvision.models.vgg import model_urls
from .pidfile import exit_if_job_done, mark_job_done

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
args = parser.parse_args()

global_seed = 1
expname = 'invert_over5_resnet'
expdir = os.path.join('results', args.model, expname)
os.makedirs(expdir, exist_ok=True)

def main():
    torch.manual_seed(global_seed)
    pbar.print('Training %s' % expdir)

    # Load a progressive GAN
    full_generator = setting.load_proggan(args.model)
    # Make a subset model with only some layers.
    decoder = nethook.subsequence(full_generator, first_layer='layer5')
    generator = nethook.subsequence(full_generator, last_layer='layer4')

    # Make an encoder model.
    encoder = encoder_net.make_over5_resnet()

    # Also make a conv features model from pretrained VGG
    vgg = models.vgg16(pretrained=True)
    features = nethook.subsequence(vgg.features, last_layer='20')

    # Move models to GPU
    for m in [generator, decoder, encoder, features]:
        m.cuda()

    # Set up adata loaders that just feed random z
    batch_size = 32
    train_loader = training_loader(generator, batch_size)
    test_loader = testing_loader(generator, batch_size)

    # Set up optimizer
    set_requires_grad(False, decoder, generator, features)
    learning_rate = args.lr
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    epoch_batches = 100
    num_epochs = 100
    # img_elems = 256*256*3
    # rep_elems = 8*8*512
    # alpha = float(rep_elems) / (rep_elems + img_elems)
    for epoch, epoch_loader in enumerate(pbar(
            epoch_grouper(train_loader, epoch_batches), total=num_epochs)):
        if epoch > num_epochs:
            break
        # Training loop
        if epoch > 0:
            for (z_batch,) in pbar(epoch_loader, total=epoch_batches):
                z_batch = z_batch.cuda()
                r_batch = generator(z_batch)
                optimizer.zero_grad()
                loss = encoder_decoder_loss(encoder, decoder, r_batch)
                loss.backward()
                pbar.post(l=loss.item())
                optimizer.step()
        # Testing loop
        with torch.no_grad():
            loss = 0.0
            count = 0
            for i, (z_batch, ) in enumerate(pbar(test_loader)):
                z_batch = z_batch.cuda()
                r_batch = generator(z_batch)
                count += len(z_batch)
                loss += (encoder_decoder_loss(encoder, decoder, r_batch) *
                        len(z_batch))
                if i == 0 and epoch % 10 == 0:
                    visualize_results(epoch, r_batch, encoder, decoder)
            loss /= count
            pbar.print("Epoch", epoch, "Loss", loss.item())
            with open(os.path.join(expdir, 'log.txt'), 'a') as f:
                f.write('{} {}\n'.format(epoch, loss.item()))
        if epoch % 10 == 0:
            save_checkpoint(
                    epoch=epoch,
                    state_dict=encoder.state_dict(),
                    loss=loss.item(),
                    lr=learning_rate,
                    optimizer=optimizer.state_dict())

def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'epoch_%d.pth.tar' % kwargs['epoch']
    torch.save(kwargs, os.path.join(dirname, filename))

def visualize_results(epoch, z_batch, encoder, decoder):
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    generated = decoder(z_batch)
    encoded = encoder(generated)
    recovered = decoder(encoded)
    for i in range(min(len(z_batch), 6)):
        save_tensor_image(generated[i], os.path.join(dirname,
            'epoch_%d_%d_g.png' % (epoch, i)))
        save_tensor_image(recovered[i], os.path.join(dirname,
            'epoch_%d_%d_r.png' % (epoch, i)))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'lightbox.html'),
            os.path.join(dirname, '+lightbox.html'))

def save_tensor_image(img, filename):
    np_data = ((img.permute(1, 2, 0) / 2 + 0.5) * 255).byte().cpu().numpy()
    PIL.Image.fromarray(np_data).save(filename)
    
def encoder_decoder_loss(encoder, decoder, encoded_batch):
    reencoded_batch = encoder(decoder(encoded_batch))
    encoder_loss = mse_loss(encoded_batch, reencoded_batch)
    return encoder_loss

def training_loader(z_generator, batch_size):
    '''
    Returns an infinite generator that runs through randomized
    training data repeatedly in shuffled order, forever.
    '''
    epoch = 0
    while True:
        z_dataset = zdataset.z_dataset_for_model(
                z_generator, size=batch_size * 50, seed=epoch + global_seed)
        dataloader = torch.utils.data.DataLoader(
                z_dataset,
                batch_size=batch_size, num_workers=2,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        epoch += 1

def testing_loader(z_generator, batch_size):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    z_dataset = zdataset.z_dataset_for_model(
        z_generator, size=1000, seed=global_seed - 1)
    dataloader = torch.utils.data.DataLoader(
            z_dataset,
            batch_size=32, num_workers=2,
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

if __name__ == '__main__':
    exit_if_job_done(expdir)
    main()
    mark_job_done(expdir)
