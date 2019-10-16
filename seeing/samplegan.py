'''
A simple tool to generate a sample of output of a GAN.
'''

import torch, numpy, os, argparse, numbers, sys, shutil
from PIL import Image
from torch.utils.data import TensorDataset
from .zdataset import standard_z_sample
from . import pbar
from .autoeval import autoimport_eval
from .workerpool import WorkerBase, WorkerPool

def main():
    parser = argparse.ArgumentParser(description='GAN sample making utility')
    parser.add_argument('--model', type=str, default=None,
            help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
            help='filename of .pth file for the model')
    parser.add_argument('--outdir', type=str, default='images',
            help='directory for image output')
    parser.add_argument('--size', type=int, default=100,
            help='number of images to output')
    parser.add_argument('--seed', type=int, default=1,
            help='seed')
    parser.add_argument('--quiet', action='store_true', default=False,
            help='silences console output')
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    pbar.quiet(args.quiet)

    # Instantiate the model
    model = autoimport_eval(args.model)
    if args.pthfile is not None:
        data = torch.load(args.pthfile)
        if 'state_dict' in data:
            data = data['state_dict']
        model.load_state_dict(data)
    # Unwrap any DataParallel-wrapped model
    if isinstance(model, torch.nn.DataParallel):
        model = next(model.children())
    # Examine first conv in model to determine input feature size.
    first_layer = [c for c in model.modules()
            if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                torch.nn.Linear))][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        z_channels = first_layer.in_channels
        spatialdims = (1, 1)
    else:
        z_channels = first_layer.in_features
        spatialdims = ()
    model.cuda()

    # Get the sample of z vectors
    indexes = torch.arange(args.size)
    z_sample = standard_z_sample(args.size, z_channels, seed=args.seed)
    z_sample = z_sample.view(tuple(z_sample.shape) + spatialdims)

    save_znum_images(args.outdir, model, z_sample, indexes)
    copy_lightbox_to(args.outdir)



def save_znum_images(dirname, model, z_sample, indexes,
        name_template="image_{}.png", lightbox=False, batch_size=100):
    os.makedirs(dirname, exist_ok=True)
    with torch.no_grad():
        # Now generate images
        z_loader = torch.utils.data.DataLoader(TensorDataset(z_sample),
                    batch_size=batch_size, num_workers=2,
                    pin_memory=True)
        saver = WorkerPool(SaveImageWorker)
        for batch_num, [z] in enumerate(pbar(z_loader,
                desc='Saving images')):
            z = z.cuda()
            start_index = batch_num * batch_size
            im = ((model(z) + 1) / 2 * 255).clamp(0, 255).byte().permute(
                    0, 2, 3, 1).cpu()
            for i in range(len(im)):
                index = i + start_index
                if indexes is not None:
                    index = indexes[index].item()
                filename = os.path.join(dirname, name_template.format(index))
                saver.add(im[i].numpy(), filename)
    saver.join()

def copy_lightbox_to(dirname):
   srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
   shutil.copy(os.path.join(srcdir, 'lightbox.html'),
           os.path.join(dirname, '+lightbox.html'))

class SaveImageWorker(WorkerBase):
    def work(self, data, filename):
        Image.fromarray(data).save(filename, optimize=True, quality=100)

if __name__ == '__main__':
    main()
