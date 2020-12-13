"""
Base implementation of slimmable SNGAN with default variables.
"""
import torch
import torch.nn.functional as F
from torch_mimicry.nets.gan import gan
from torch_mimicry.modules.slimmable_ops import FLAGS


class SlimmableSNGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for slimmable SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', alpha=0, stepwise=False,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.alpha = alpha
        self.stepwise = stepwise

    def compute_distill_loss(self, images, stepwise=False):
        loss = 0
        for i in range(0, len(images)-1):
            loss += F.mse_loss(images[i], images[i+1 if stepwise else -1].detach())
        return loss / (len(images)-1)

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        """
        Train step function.
        """

        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        errG = []
        images = []
        fixed_noise = torch.randn((batch_size, self.nz), device=device)
        for i, width_mult in enumerate(FLAGS.width_mult_list):
            FLAGS.width_mult = width_mult
            # Produce fake images and logits
            fake_images = self.generate_images(num_images=batch_size,
                                               device=device)
            output = netD(fake_images)

            if self.alpha > 0:
                fake_images = self.generate_images(num_images=batch_size,
                                                   noise=fixed_noise,
                                                   device=device)
                images.append(fake_images)

            # Compute GAN loss, upright images only.
            errG.append(self.compute_gan_loss(output))

            # Backprop and update gradients
            errG_total = errG[-1]
            errG_total.backward()
            log_data.add_metric('errG' + str(i), errG[-1], group='loss')

        if self.alpha > 0:
            lossDistill = self.compute_distill_loss(images, self.stepwise) * self.alpha
            lossDistill.backward()
            log_data.add_metric('lossDistill', lossDistill, group='lossDistill')

        optG.step()
            
        return log_data


class SlimmableSNGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, ndf, loss_type='hinge', n_share=0, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.n_share = n_share

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        """
        Train step function for discirminator.
        """
        self.zero_grad()

        # Produce real images
        real_images, _ = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        errD = []
        for width_mult in FLAGS.width_mult_list:
            FLAGS.width_mult = width_mult
            # Produce fake images
            fake_images = netG.generate_images(num_images=batch_size,
                                               device=device).detach()

            # Compute real and fake logits for gan loss
            output_real = self.forward(real_images)
            output_fake = self.forward(fake_images)

            # Compute GAN loss, upright images only.
            errD.append(self.compute_gan_loss(output_real=output_real,
                                              output_fake=output_fake))

            # Backprop and update gradients
            errD_total = errD[-1]
            errD_total.backward()

        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        for i, err in enumerate(errD):
            log_data.add_metric('errD' + str(i), err, group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
