import torch, torchvision, os
from . import parallelfolder, zdataset

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth')[domain]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

def load_proggan_ablation(modelname):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)

    from . import proggan_ablation
    model_classname, weights_filename = {
		"equalized-learning-rate": (proggan_ablation.G128_equallr,
            "equalized-learning-rate-88ed833d.pth"),
        "minibatch-discrimination": (proggan_ablation.G128_minibatch_disc,
            "minibatch-discrimination-604c5731.pth"),
        "minibatch-stddev": (proggan_ablation.G128_minibatch_disc,
            "minibatch-stddev-068bc667.pth"),
        "pixelwise-normalization": (proggan_ablation.G128_pixelwisenorm,
            "pixelwise-normalization-4da7e9ce.pth"),
        "progressive-training": (proggan_ablation.G128_simple,
            "progressive-training-70bd90ac.pth"),
        # "revised-training-parameters": (_,
        #     "revised-training-parameters-902f5486.pth")
        "small-minibatch": (proggan_ablation.G128_simple,
            "small-minibatch-04143d18.pth"),
        "wgangp": (proggan_ablation.G128_simple,
            "wgangp-beaa509a.pth")
        }[modelname]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/ablations/' + weights_filename
    # try:
    sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    #except:
    #    sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = model_classname()
    model.load_state_dict(sd)
    return model

g_datasets = {}

def load_dataset(domain, split='train', full=False, download=True):
    if domain in g_datasets:
        return g_datasets[domain]
    dirname = os.path.join('datasets', 'lsun' if full else 'minilsun', domain)
    dirname += '_' + split
    if download and not full and not os.path.exists('datasets/minilsun'):
        torchvision.datasets.utils.download_and_extract_archive(
                'http://gandissect.csail.mit.edu/datasets/minilsun.zip',
                'datasets',
                md5='a67a898673a559db95601314b9b51cd5')
    return parallelfolder.ParallelImageFolders([dirname],
            transform=g_transform)

g_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_test_image(imgnum, split, model, full=False):
    if split == 'gan':
        with torch.no_grad():
            generator = load_proggan(model)
            z = zdataset.z_sample_for_model(generator, size=(imgnum + 1)
                    )[imgnum]
            z = z[None]
            return generator(z), z
    assert split in ['train', 'val']
    ds = load_dataset(model, split, full=full)
    return ds[imgnum][0][None], None

if __name__ == '__main__':
    main()

