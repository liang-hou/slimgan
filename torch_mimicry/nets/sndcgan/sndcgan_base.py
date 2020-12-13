"""
Base implementation of SNDCGAN with default variables.
"""
from torch_mimicry.nets.gan import gan


class SNDCGANBaseGenerator(gan.BaseGenerator):
    r"""
    Deep Convolution backbone generator for SNDCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)


class SNDCGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    Deep Convolution backbone discriminator for SNDCGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, ndf, loss_type='hinge', **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
