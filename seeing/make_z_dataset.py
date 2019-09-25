import torch, copy, argparse, re, os, numpy
from seeing import nethook, setting, renormalize, zdataset, pbar
from seeing import encoder_net, nethook
from seeing import workerpool
from seeing.encoder_loss import cor_square_error
from torch.nn.functional import mse_loss, l1_loss
from seeing.LBFGS import FullBatchLBFGS


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='church')
parser.add_argument('--dataset', default='church_outdoor_train')
parser.add_argument('--iterations', type=int, default=0)
args = parser.parse_args()

batch_size = 32
expdir = 'results/z_dataset/%s/it_%d' % (args.dataset, args.iterations)

G = setting.load_proggan('church').eval().cuda()

E = encoder_net.HybridLayerNormEncoder()
filename = 'results/church/invert_hybrid_bottom_b5/snapshots/epoch_1000.pth.tar'
E.load_state_dict(torch.load(filename)['state_dict'])
E.eval().cuda()

dataset = setting.load_dataset(args.dataset, full=True)
loader = torch.utils.data.DataLoader(dataset, shuffle=False,
        batch_size=batch_size, num_workers=10, pin_memory=True)

torch.set_grad_enabled(False)

class SaveNpyWorker(workerpool.WorkerBase):
    def work(self, filename, data):
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        numpy.save(filename, data)

saver_pool = workerpool.WorkerPool(worker=SaveNpyWorker)

def target_filename_from_source(filename):
    patharr = filename.split('/')
    patharr = patharr[patharr.index(args.dataset)+1:]
    patharr[-1] = os.path.splitext(patharr[-1])[0] + '.npy'
    return os.path.join(expdir, *patharr)

def refine_z_lbfgs(init_z, target_x, lambda_f=1.0, num_steps=100):

    z = init_z.clone()
    parameters = [z]
    F = E
    target_f = F(target_x)
    nethook.set_requires_grad(False, G, E, target_x, target_f)
    nethook.set_requires_grad(True, *parameters)
    optimizer = FullBatchLBFGS(parameters)

    def closure():
        optimizer.zero_grad()
        current_x = G(z)
        loss = l1_loss(target_x, current_x)
        if lambda_f:
            loss += mse_loss(target_f, F(current_x)) * lambda_f
        return loss

    with torch.enable_grad():
        for step_num in pbar(range(num_steps + 1)):
            if step_num == 0:
                loss = closure()
                loss.backward()
            else:
                options = dict(closure=closure, current_loss=loss, max_ls=10)
                loss, _, _, _, _, _, _, _ = optimizer.step(options)
    return z

index = 0
for [im] in pbar(loader):
    im = im.cuda()
    z = E(im)
    if args.iterations > 0:
        #for i in range(len(im)):
        #    z[i:i+1] = refine_z_lbfgs(z[i:i+1], im[i:i+1],
        #            num_steps=args.iterations)
        z = refine_z_lbfgs(z, im, num_steps=args.iterations)

    cpu_z = z.cpu().numpy()
    for i in range(len(im)):
        filename = target_filename_from_source(dataset.images[index + i][0])
        data = cpu_z[i].copy()[None]
        pbar.print(filename, data.shape)
        saver_pool.add(filename, data)
    index += len(im)
saver_pool.join()
