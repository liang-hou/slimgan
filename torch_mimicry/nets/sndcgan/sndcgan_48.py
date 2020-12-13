"""
Implementation of SNDCGAN for image size 48.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear, SNConv2d
from torch_mimicry.nets.sndcgan import sndcgan_base


class SNDCGANGenerator48(sndcgan_base.SNDCGANBaseGenerator):
    r"""
    Deep Convolution backbone generator for SNDCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=512, bottom_width=6, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.c1 = nn.ConvTranspose2d(self.ngf, self.ngf >> 1, 4, 2, 1)
        self.b1 = nn.BatchNorm2d(self.ngf >> 1)
        self.c2 = nn.ConvTranspose2d(self.ngf >> 1, self.ngf >> 2, 4, 2, 1)
        self.b2 = nn.BatchNorm2d(self.ngf >> 2)
        self.c3 = nn.ConvTranspose2d(self.ngf >> 2, self.ngf >> 3, 4, 2, 1)
        self.b3 = nn.BatchNorm2d(self.ngf >> 3)
        self.c4 = nn.Conv2d(self.ngf >> 3, 3, 3, 1, 1)
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


class SNDCGANDiscriminator48(sndcgan_base.SNDCGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=512, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.c1 = SNConv2d(3, self.ndf >> 3, 3, 1, 1)
        self.c2 = SNConv2d(self.ndf >> 3, self.ndf >> 3, 4, 2, 1)

        self.c3 = SNConv2d(self.ndf >> 3, self.ndf >> 2, 3, 1, 1)
        self.c4 = SNConv2d(self.ndf >> 2, self.ndf >> 2, 4, 2, 1)

        self.c5 = SNConv2d(self.ndf >> 2, self.ndf >> 1, 3, 1, 1)
        self.c6 = SNConv2d(self.ndf >> 1, self.ndf >> 1, 4, 2, 1)

        self.c7 = SNConv2d(self.ndf >> 1, self.ndf, 3, 1, 1)

        self.l8 = SNLinear(ndf * 6 * 6, 1)

        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.c1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c3.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c4.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l8.weight.data, 1.0)

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

        h = h.view(h.shape[0], -1)
        output = self.l8(h)

        return output