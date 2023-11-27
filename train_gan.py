import os
import argparse
from data_loader import get_loader

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.backends import cudnn

import numpy as np
import random

from models import Discriminator64, Generator64, SNDiscriminator64, SNGenerator64
from utils import *

def main(config):
    # For fast training.
    cuda = True if torch.cuda.is_available() else False
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader.
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)

    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    if config.arch == 'DCGAN':
        generator = Generator64()
        discriminator = Discriminator64()
    elif config.arch == 'SNDCGAN':
        generator = SNGenerator64()
        discriminator = SNDiscriminator64()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr/3, betas=(config.b1, config.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    dataloader = celeba_loader

    # ----------
    #  Training
    # ----------

    for epoch in range(config.n_epochs):
        for i, (imgs, _) in enumerate (dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config.latent_dim))))
            # Generate a batch of images
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            feat, out = discriminator(gen_imgs)
            z_corr = pearson(z, feat)

            g_loss = adversarial_loss(out, valid) - 0.5 * torch.mean(z_corr)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            feat_real, out_real = discriminator(real_imgs)
            feat_fake, out_fake = discriminator(gen_imgs.detach())

            real_loss = adversarial_loss(out_real, valid)
            fake_loss = adversarial_loss(out_fake, fake)
            z_corr = pearson(z, feat_fake)

            d_loss = (real_loss + fake_loss) / 2 - 0.5*torch.mean(z_corr)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, config.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % config.sample_interval == 0:
                save_image(gen_imgs.data[:25], "./images/%d.png" % batches_done, nrow=5, normalize=True)
                print("save images.")

    torch.save(generator.state_dict(), os.path.join(config.model_save_dir, "G_"+config.arch+config.dataset+str(config.seed)+".pth"))
    torch.save(discriminator.state_dict(), os.path.join(config.model_save_dir, "D_"+config.arch+config.dataset+str(config.seed)+".pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")

    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")

    # Model configuration.
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=64, help='image resolution')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--mode', type=str, default='train')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='./data/img_align_celeba')
    parser.add_argument('--attr_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_path', type=str, default='./images')

    parser.add_argument('--seed', type=int, default=3407, help='Set seed. If -1, use randomization.')
    parser.add_argument('--arch', type=str, default='DCGAN', help='model architecture')

    config = parser.parse_args()
    print(config)

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if config.seed != -1:
        random.seed(config.seed)
        np.random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        print('Set seed to %d.' % config.seed)

    main(config)