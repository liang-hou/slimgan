"""
PyTorch interface for computing mIC.
"""
import os
import random
import time

import numpy as np
import tensorflow as tf
import torch

from torch_mimicry.metrics.inception_model import inception_utils

from torch_mimicry.modules.resblocks import FLAGS


def _normalize_images(images):
    """
    Given a tensor of images, uses the torchvision
    normalization method to convert floating point data to integers. See reference
    at: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image

    The function uses the normalization from make_grid and save_image functions.

    Args:
        images (Tensor): Batch of images of shape (N, 3, H, W).

    Returns:
        ndarray: Batch of normalized images of shape (N, H, W, 3).
    """
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    images = images.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(
        'cpu', torch.uint8).numpy()

    return images


def compute_gen_dist_stats(netG,
                           num_samples,
                           width_mult_idx,
                           fixed_noise,
                           fixed_class_labels,
                           sess,
                           device,
                           seed,
                           batch_size,
                           print_every=20,
                           verbose=True):
    """
    Directly produces the images and convert them into numpy format without
    saving the images on disk.

    Args:
        netG (Module): Torch Module object representing the generator model.
        num_samples (int): The number of fake images for computing statistics.
        sess (Session): TensorFlow session to use.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        batch_size (int): The number of samples per batch for inference.
        print_every (int): Interval for printing log.
        verbose (bool): If True, prints progress.

    Returns:
        ndarray: Mean features stored as np array.
        ndarray: Covariance of features stored as np array.
    """
    with torch.no_grad():
        # Set model to evaluation mode
        netG.eval()

        # Inference variables
        batch_size = min(num_samples, batch_size)

        # Collect all samples()
        images = []
        start_time = time.time()

        FLAGS.width_mult = FLAGS.width_mult_list[width_mult_idx]
        for idx in range(num_samples // batch_size):
            # Collect fake image
            if hasattr(netG, 'num_classes') and netG.num_classes > 0:
                fake_images = netG.generate_images_with_labels(num_images=batch_size,
                                                   noise=fixed_noise[idx],
                                                   fake_class_labels=fixed_class_labels[idx],
                                                   device=device)[0].detach().cpu()
            else:
                fake_images = netG.generate_images(num_images=batch_size,
                                                   noise=fixed_noise[idx],
                                                   device=device).detach().cpu()
            images.append(fake_images)

            # Print some statistics
            if (idx + 1) % print_every == 0:
                end_time = time.time()
                print(
                    "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                    .format(
                        (idx + 1) * batch_size, num_samples, seed,
                        (end_time - start_time) / (print_every * batch_size)))
                start_time = end_time

        # Produce images in the required (N, H, W, 3) format for FID computation
        images = torch.cat(images, 0)  # Gives (N, 3, H, W)
        images = _normalize_images(images)  # Gives (N, H, W, 3)

    # Compute the features
    act = inception_utils.get_activations(images, sess, batch_size, verbose)
    return act


def mic_score(num_samples,
              netG,
              device,
              seed,
              batch_size=50,
              verbose=True,
              log_dir='./log'):
    """
    Computes mIC stats using functions that store images in memory for speed and fidelity.
    Fidelity since by storing images in memory, we don't subject the scores to different read/write
    implementations of imaging libraries.

    Args:
        num_samples (int): The number of real images to use for mIC.
        netG (Module): Torch Module object representing the generator model.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        dataset_name (str): The name of the dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        verbose (bool): If True, prints progress.
        stats_file (str): The statistics file to load from if there is already one.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        float: Scalar mIC score.
    """
    start_time = time.time()

    # Make sure the random seeds are fixed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Setup directories
    inception_path = os.path.join('metrics', 'inception_model')

    # Setup the inception graph
    inception_utils.create_inception_graph(inception_path)

    # Start producing statistics for real and fake images
    if device and device.index is not None:
        # Avoid unbounded memory usage
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction=0.15,
                                    visible_device_list=str(device.index))
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    else:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        acts = []

        fixed_noise = []
        for idx in range(num_samples // batch_size):
            fixed_noise.append(torch.randn((batch_size, netG.nz), device=device))

        fixed_class_labels = None
        if hasattr(netG, 'num_classes') and netG.num_classes > 0:
            fixed_class_labels = []
            for idx in range(num_samples // batch_size):
                fixed_class_labels.append(torch.randint(low=0, high=netG.num_classes,
                                          size=(batch_size,),
                                          device=device))

        for width_mult_idx in range(len(FLAGS.width_mult_list)):
            acts.append(compute_gen_dist_stats(netG=netG,
                                               num_samples=num_samples,
                                               width_mult_idx=width_mult_idx,
                                               fixed_noise=fixed_noise,
                                               fixed_class_labels=fixed_class_labels,
                                               sess=sess,
                                               device=device,
                                               seed=seed,
                                               batch_size=batch_size,
                                               verbose=verbose))

        mIC_scores = []
        for i in range(0, len(acts)):
            for j in range(0, len(acts)):
                if i == j:
                    continue
                mIC_scores.append((((acts[i] - acts[j]) ** 2).sum(1)).mean())
        mIC_score = np.mean(mIC_scores)

        print("INFO: mIC Score: {} [Time Taken: {:.4f} secs]".format(
            mIC_score,
            time.time() - start_time))

        return float(mIC_score)
