"""
Implementation of SNDCGAN for image size 64.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear, SNConv2d
from torch_mimicry.nets.sndcgan import sndcgan_base


class SNDCGANGenerator64(sndcgan_base.SNDCGANBaseGenerator):
    r"""
    Deep Convolution backbone generator for SNDCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.c1 = nn.ConvTranspose2d(self.ngf, self.ngf >> 1, 4, 2, 1)
        self.b1 = nn.BatchNorm2d(self.ngf >> 1)
        self.c2 = nn.ConvTranspose2d(self.ngf >> 1, self.ngf >> 2, 4, 2, 1)
        self.b2 = nn.BatchNorm2d(self.ngf >> 2)
        self.c3 = nn.ConvTranspose2d(self.ngf >> 2, self.ngf >> 3, 4, 2, 1)
        self.b3 = nn.BatchNorm2d(self.ngf >> 3)
        self.c4 = nn.ConvTranspose2d(self.ngf >> 3, self.ngf >> 4, 4, 2, 1)
        self.b4 = nn.BatchNorm2d(self.ngf >> 4)
        self.c5 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c3.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c4.weight.data, 1.0)
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
        h = self.activation(self.b1(self.c1(h)))
        h = self.activation(self.b2(self.c2(h)))
        h = self.activation(self.b3(self.c3(h)))
        h = self.activation(self.b4(self.c4(h)))
        h = torch.tanh(self.c5(h))

        return h


class SNDCGANDiscriminator64(sndcgan_base.SNDCGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.c1 = SNConv2d(3, self.ndf >> 4, 3, 1, 1)
        self.c2 = SNConv2d(self.ndf >> 4, self.ndf >> 4, 4, 2, 1)

        self.c3 = SNConv2d(self.ndf >> 4, self.ndf >> 3, 3, 1, 1)
        self.c4 = SNConv2d(self.ndf >> 3, self.ndf >> 3, 4, 2, 1)

        self.c5 = SNConv2d(self.ndf >> 3, self.ndf >> 2, 3, 1, 1)
        self.c6 = SNConv2d(self.ndf >> 2, self.ndf >> 2, 4, 2, 1)

        self.c7 = SNConv2d(self.ndf >> 2, self.ndf >> 1, 3, 1, 1)
        self.c8 = SNConv2d(self.ndf >> 1, self.ndf >> 1, 4, 2, 1)

        self.c9 = SNConv2d(self.ndf >> 1, self.ndf, 3, 1, 1)

        self.l10 = SNLinear(ndf * 4 * 4, 1)

        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.c1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c3.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c4.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c8.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c9.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l10.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.activation(self.c1(h))
        h = self.activation(self.c2(h))

        h = self.activation(self.c3(h))
        h = self.activation(self.c4(h))

        h = self.activation(self.c5(h))
        h = self.activation(self.c6(h))

        h = self.activation(self.c7(h))
        h = self.activation(self.c8(h))

        h = self.activation(self.c9(h))

        h = h.view(h.shape[0], -1)
        output = self.l10(h)

        return output