import sys
sys.path.append('..')

import torch
import torch.optim as optim
import torch_mimicry as mmc

from torch_mimicry.nets.sngan import \
    SNGANGenerator32, SNGANDiscriminator32, \
    SNGANGenerator48, SNGANDiscriminator48, \
    SNGANGenerator64, SNGANDiscriminator64

from torch_mimicry.nets.sndcgan import \
    SNDCGANGenerator32, SNDCGANDiscriminator32, \
    SNDCGANGenerator48, SNDCGANDiscriminator48, \
    SNDCGANGenerator64, SNDCGANDiscriminator64

from torch_mimicry.nets.cgan_pd import CGANPDGenerator32, CGANPDDiscriminator32

import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--dataset_dir', type=str, default='./datasets')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'stl10_48', 'celeba_64'])
parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'dcgan'])
parser.add_argument('--width_mult_g', type=float, default=1.0, choices=[0.25, 0.5, 0.75, 1.0])
parser.add_argument('--width_mult_d', type=float, default=1.0, choices=[0.25, 0.5, 0.75, 1.0])
parser.add_argument('--log_dir', type=str, default='./logs/baseline')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--setting', type=str, default='', choices=['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
parser.add_argument('--lr_d', type=float, default=2e-4)
parser.add_argument('--lr_g', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--n_dis', type=int, default=5)
parser.add_argument('--n_steps', type=int, default=100000)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--cond', action='store_true', default=False)

args = parser.parse_args()

settings = {
    'A': {'lr': 1e-4, 'beta1': 0.5, 'beta2': 0.9, 'n_dis':5},
    'B': {'lr': 1e-4, 'beta1': 0.5, 'beta2': 0.999, 'n_dis':1},
    'C': {'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999, 'n_dis':1},
    'D': {'lr': 1e-3, 'beta1': 0.5, 'beta2': 0.9, 'n_dis':5},
    'E': {'lr': 1e-3, 'beta1': 0.5, 'beta2': 0.999, 'n_dis':5},
    'F': {'lr': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'n_dis':5},
    'G': {'lr': 2e-4, 'beta1': 0.0, 'beta2': 0.9, 'n_dis':5},
}

if args.setting:
    args.lr_d = args.lr_g = settings[args.setting]['lr']
    args.beta1 = settings[args.setting]['beta1']
    args.beta2 = settings[args.setting]['beta2']
    args.n_dis = settings[args.setting]['n_dis']

if args.dataset in ['cifar10', 'cifar100']:
    if args.arch == 'resnet':
        args.ngf = 256
        args.ndf = 128
    elif args.arch == 'dcgan':
        args.ngf = 512
        args.ndf = 512
elif args.dataset == 'stl10_48':
    if args.arch == 'resnet':
        args.ngf = 512
        args.ndf = 1024
    elif args.arch == 'dcgan':
        args.ngf = 512
        args.ndf = 512
elif args.dataset == 'celeba_64':
    if args.arch == 'resnet':
        args.ngf = 1024
        args.ndf = 1024
    elif args.arch == 'dcgan':
        args.ngf = 1024
        args.ndf = 1024

if __name__ == "__main__":
    # Fast training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root=args.dataset_dir, name=args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define models and optimizers
    Generator = None
    Discriminator = None
    if args.cond:
        assert args.dataset in ['cifar10', 'cifar100']
        Generator = CGANPDGenerator32
        Discriminator = CGANPDDiscriminator32
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        netG = Generator(num_classes=num_classes, ngf=int(args.ngf * args.width_mult_g), loss_type=args.loss).to(device)
        netD = Discriminator(num_classes=num_classes, ndf=int(args.ndf * args.width_mult_d), loss_type=args.loss).to(device)
    else:
        if args.arch == 'resnet':
            if args.dataset in ['cifar10', 'cifar100']:
                Generator = SNGANGenerator32
                Discriminator = SNGANDiscriminator32
            elif args.dataset == 'stl10_48':
                Generator = SNGANGenerator48
                Discriminator = SNGANDiscriminator48
            elif args.dataset == 'celeba_64':
                Generator = SNGANGenerator64
                Discriminator = SNGANDiscriminator64
        elif args.arch == 'dcgan':
            if args.dataset in ['cifar10', 'cifar100']:
                Generator = SNDCGANGenerator32
                Discriminator = SNDCGANDiscriminator32
            elif args.dataset == 'stl10_48':
                Generator = SNDCGANGenerator48
                Discriminator = SNDCGANDiscriminator48
            elif args.dataset == 'celeba_64':
                Generator = SNDCGANGenerator64
                Discriminator = SNDCGANDiscriminator64
        netG = Generator(ngf=int(args.ngf * args.width_mult_g), loss_type=args.loss).to(device)
        netD = Discriminator(ndf=int(args.ndf * args.width_mult_d), loss_type=args.loss).to(device)
        
    optD = optim.Adam(netD.parameters(), args.lr_d, betas=(args.beta1, args.beta2))
    optG = optim.Adam(netG.parameters(), args.lr_g, betas=(args.beta1, args.beta2))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=args.n_dis,
        num_steps=args.n_steps,
        dataloader=dataloader,
        log_dir=args.log_dir,
        device=device
    )
    trainer.train()

    num_real_samples = 10000
    num_fake_samples = 10000
    num_samples = 50000
    if args.dataset == 'stl10_48':
        num_real_samples = 8000
        num_fake_samples = 8000

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir=args.log_dir,
        netG=netG,
        dataset=args.dataset,
        num_real_samples=num_real_samples,
        num_fake_samples=num_fake_samples,
        evaluate_step=args.n_steps,
        device=device,
        split='test')

    # Evaluate inception score
    mmc.metrics.evaluate(
        metric='inception_score',
        log_dir=args.log_dir,
        netG=netG,
        num_samples=num_samples,
        evaluate_step=args.n_steps,
        device=device)
