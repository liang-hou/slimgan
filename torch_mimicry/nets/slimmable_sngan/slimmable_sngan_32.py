"""
Implementation of slimmable slimmable SNGAN for image size 32.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear, SNConv2d
from torch_mimicry.modules.slimmable_ops import FLAGS, SwitchableBatchNorm2d, \
    SlimmableLinear, SlimmableConv2d, \
    SNSlimmableLinear, SNSlimmableConv2d
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock, SlimmableGBlock
from torch_mimicry.nets.slimmable_sngan import slimmable_sngan_base


class SlimmableSNGANGenerator32(slimmable_sngan_base.SlimmableSNGANBaseGenerator):
    r"""
    ResNet backbone generator for slimmable SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=256, bottom_width=4, alpha=0, stepwise=False,**kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width,
                         alpha=alpha, stepwise=stepwise, **kwargs)

        # Build the layers
        self.l1 = SlimmableLinear([self.nz for _ in FLAGS.width_mult_list], [int((self.bottom_width ** 2) * self.ngf * width_mult) for width_mult in FLAGS.width_mult_list])
        self.block2 = SlimmableGBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = SlimmableGBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = SlimmableGBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = SwitchableBatchNorm2d([int(self.ngf * width_mult) for width_mult in FLAGS.width_mult_list])
        self.c5 = SlimmableConv2d([int(self.ngf * width_mult) for width_mult in FLAGS.width_mult_list], [3 for _ in FLAGS.width_mult_list], 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))

        return h


class SlimmableSNGANDiscriminator32(slimmable_sngan_base.SlimmableSNGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for slimmable SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=128, n_share=0, **kwargs):
        super().__init__(ndf=ndf, n_share=n_share, **kwargs)

        assert -1 <= self.n_share <= 4

        # # Build layers

        block1s = []
        block2s = []
        block3s = []
        block4s = []
        l5s = []

        for width_mult in FLAGS.width_mult_list:
            block1s.append(DBlockOptimized(3, self.ndf))
            block2s.append(DBlock(self.ndf, self.ndf, downsample=True))
            block3s.append(DBlock(self.ndf, self.ndf, downsample=False))
            block4s.append(DBlock(self.ndf, self.ndf, downsample=False))
            l5s.append(SNLinear(self.ndf, 1))

        self.block1s = nn.ModuleList(block1s)
        self.block2s = nn.ModuleList(block2s)
        self.block3s = nn.ModuleList(block3s)
        self.block4s = nn.ModuleList(block4s)
        self.l5s = nn.ModuleList(l5s)

        self.activation = nn.ReLU(True)

        # Initialise the weights
        for l in self.l5s:
            nn.init.xavier_uniform_(l.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """

        idx = int(FLAGS.width_mult / 0.25) - 1

        h = x
        h = self.block1s[-1 if self.n_share > 0 else idx](h)
        h = self.block2s[-1 if self.n_share > 1 else idx](h)
        h = self.block3s[-1 if self.n_share > 2 else idx](h)
        h = self.block4s[-1 if self.n_share > 3 else idx](h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))

        output = self.l5s[-1 if self.n_share == -1 else idx](h)

        return output
