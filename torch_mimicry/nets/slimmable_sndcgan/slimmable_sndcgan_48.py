"""
Implementation of SNDCGAN for image size 48.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear, SNConv2d
from torch_mimicry.modules.slimmable_ops import FLAGS, SwitchableBatchNorm2d, \
    SlimmableLinear, SlimmableConvTranspose2d, SlimmableConv2d, \
    SNSlimmableLinear, SNSlimmableConv2d
from torch_mimicry.nets.slimmable_sndcgan import slimmable_sndcgan_base


class SlimmableSNDCGANGenerator48(slimmable_sndcgan_base.SlimmableSNDCGANBaseGenerator):
    r"""
    DCGAN backbone generator for slimmable SNDCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=512, bottom_width=6,
                 alpha=0, stepwise=False, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width,
                         alpha=alpha, stepwise=stepwise, **kwargs)

        # Build the layers
        self.l1 = SlimmableLinear(
            [self.nz for width_mult in FLAGS.width_mult_list],
            [int((self.bottom_width**2) * self.ngf * width_mult) for width_mult in FLAGS.width_mult_list]
        )
        self.c1 = SlimmableConvTranspose2d(
            [int(self.ngf * width_mult) for width_mult in FLAGS.width_mult_list],
            [int(self.ngf * width_mult) >> 1 for width_mult in FLAGS.width_mult_list],
             4, 2, 1
        )
        self.b1 = SwitchableBatchNorm2d([int(self.ngf * width_mult) >> 1 for width_mult in FLAGS.width_mult_list])

        self.c2 = SlimmableConvTranspose2d(
            [int(self.ngf * width_mult) >> 1 for width_mult in FLAGS.width_mult_list],
            [int(self.ngf * width_mult) >> 2 for width_mult in FLAGS.width_mult_list],
            4, 2, 1
        )
        self.b2 = SwitchableBatchNorm2d([int(self.ngf * width_mult) >> 2 for width_mult in FLAGS.width_mult_list])

        self.c3 = SlimmableConvTranspose2d(
            [int(self.ngf * width_mult) >> 2 for width_mult in FLAGS.width_mult_list],
            [int(self.ngf * width_mult) >> 3 for width_mult in FLAGS.width_mult_list],
            4, 2, 1
        )
        self.b3 = SwitchableBatchNorm2d([int(self.ngf * width_mult) >> 3 for width_mult in FLAGS.width_mult_list])

        self.c4 = SlimmableConv2d(
            [int(self.ngf * width_mult) >> 3 for width_mult in FLAGS.width_mult_list],
            [3 for width_mult in FLAGS.width_mult_list],
            3, 1, 1
        )

        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c3.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c4.weight.data, 1.0)

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
        h = self.activation(self.b1(self.c1(h)))
        h = self.activation(self.b2(self.c2(h)))
        h = self.activation(self.b3(self.c3(h)))
        h = torch.tanh(self.c4(h))

        return h


class SlimmableSNDCGANDiscriminator48(slimmable_sndcgan_base.SlimmableSNDCGANBaseDiscriminator):
    r"""
    DCGAN backbone discriminator for SNDCGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=512, n_share=0, **kwargs):
        super().__init__(ndf=ndf, n_share=n_share, **kwargs)

        assert -1 <= self.n_share <= 7

        # Build layers

        self.c1s = nn.ModuleList([
            SNConv2d(3, self.ndf >> 3, 3, 1, 1)
            for width_mult in FLAGS.width_mult_list
        ])
        self.c2s = nn.ModuleList([
            SNConv2d(self.ndf >> 3, self.ndf >> 3, 4, 2, 1)
            for width_mult in FLAGS.width_mult_list
        ])

        self.c3s = nn.ModuleList([
            SNConv2d(self.ndf >> 3, self.ndf >> 2, 3, 1, 1)
            for width_mult in FLAGS.width_mult_list
        ])
        self.c4s = nn.ModuleList([
            SNConv2d(self.ndf >> 2, self.ndf >> 2, 4, 2, 1)
            for width_mult in FLAGS.width_mult_list
        ])

        self.c5s = nn.ModuleList([
            SNConv2d(self.ndf >> 2, self.ndf >> 1, 3, 1, 1)
            for width_mult in FLAGS.width_mult_list
        ])
        self.c6s = nn.ModuleList([
            SNConv2d(self.ndf >> 1, self.ndf >> 1, 4, 2, 1)
            for width_mult in FLAGS.width_mult_list
        ])

        self.c7s = nn.ModuleList([
            SNConv2d(self.ndf >> 1, self.ndf, 3, 1, 1)
            for width_mult in FLAGS.width_mult_list
        ])

        self.l8s = nn.ModuleList([
            SNLinear(self.ndf * 6 * 6, 1)
            for width_mult in FLAGS.width_mult_list
        ])

        self.activation = nn.LeakyReLU(0.1, inplace=True)

        for i in range(len(FLAGS.width_mult_list)):
            nn.init.xavier_uniform_(self.c1s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c2s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c3s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c4s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c5s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c6s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.c7s[i].weight.data, 1.0)
            nn.init.xavier_uniform_(self.l8s[i].weight.data, 1.0)

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

        h = self.activation(self.c1s[-1 if self.n_share > 0 else idx](h))
        h = self.activation(self.c2s[-1 if self.n_share > 1 else idx](h))
        h = self.activation(self.c3s[-1 if self.n_share > 2 else idx](h))
        h = self.activation(self.c4s[-1 if self.n_share > 3 else idx](h))
        h = self.activation(self.c5s[-1 if self.n_share > 4 else idx](h))
        h = self.activation(self.c6s[-1 if self.n_share > 5 else idx](h))
        h = self.activation(self.c7s[-1 if self.n_share > 6 else idx](h))

        h = h.view(h.shape[0], -1)

        output = self.l8s[-1 if self.n_share == -1 else idx](h)

        return output
