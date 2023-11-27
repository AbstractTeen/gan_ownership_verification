import os
import argparse
from data_loader import get_loader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.backends import cudnn

import numpy as np
from models import *
from utils import *




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--G_path", type=str, default='./checkpoints/G_DCGANCelebA3407.pth', help="number of g")
    parser.add_argument("--save_img_path", type=str, default='./data/train/imgs', help="path for saving generated imgs")
    parser.add_argument("--img_nums", type=int, default=200, help="genearte nums*100 imgs")

    config = parser.parse_args()
    print(config)

    cuda = True if torch.cuda.is_available() else False
    cudnn.benchmark = True

    if not os.path.exists(config.save_img_path):
        os.makedirs(config.save_img_path)

    #fake
    generator = Generator64()
    if cuda:
        generator.cuda()

    g_dict = torch.load(config.G_path, map_location=lambda storage, loc: storage.cuda())
    generator.load_state_dict(g_dict, strict=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i in range(config.img_nums):

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (100, 128))))
        # Generate a batch of images
        gen_imgs = generator(z)

        for j in range(gen_imgs.shape[0]):
            save_image(gen_imgs.data[j], config.save_img_path + "%d.png" % (i*100+j+1), normalize=True)

        print("%d-th 100 images finished." % i)