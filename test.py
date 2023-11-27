import argparse
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets
import torchvision.transforms as transforms

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from models import Discriminator_oneclass64, Generator64 , SNGenerator64
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--objective', default='soft-boundary', choices=['one-class', 'soft-boundary'])

parser.add_argument('--source_model', type=str, default='./checkpoints/G_DCGANCelebA3407.pth')
parser.add_argument('--suspect_model', type=str, default='./checkpoints/G_SNDCGANCelebA1111.tar')
parser.add_argument('--D_new_path', type=str, default='./checkpoints/d_new.tar')

args = parser.parse_args()

def main(args):

    # load state dict
    D = Discriminator_oneclass64()
    if not args.cpu:
        D.cuda()

    d_state_dict = torch.load(args.D_new_path, map_location='cpu')

    # load one-class discrimnator
    R = d_state_dict['R']
    c = d_state_dict['c'].to("cuda")
    D.load_state_dict(d_state_dict['net_dict'], strict=True)

    # load generator
    generator = Generator64()
    generator2 = SNGenerator64()
    generator.cuda()
    generator2.cuda()
    g_dict = torch.load(args.source_model, map_location=lambda storage, loc: storage.cuda())
    g_dict2 = torch.load(args.suspect_model, map_location=lambda storage, loc: storage.cuda())
    generator.load_state_dict(g_dict, strict=True)
    generator2.load_state_dict(g_dict2, strict=True)

    Tensor = torch.cuda.FloatTensor


    auc_list = []
    for _ in range(10):
        # load test dataset
        for i in range(50):
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (10, 128))))
            z2 = Variable(Tensor(np.random.normal(0, 1, (10, 128))))
            # Generate a batch of images
            gen_imgs = generator(z)
            gen_imgs2 = generator2(z2)
            for j in range(gen_imgs.shape[0]):
                save_image(gen_imgs.data[j], "./data/test/0_source/%d.png" % (i * 10 + j + 1), normalize=True)
            for j in range(gen_imgs2.shape[0]):
                save_image(gen_imgs2.data[j], "./data/test/1_suspect/%d.png" % (i * 10 + j + 1), normalize=True)
        print("%d-th 100 images finished." % i)

        test_dataset = datasets.ImageFolder(
            './data/test',
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        auc = test(args, D, test_loader, c, R)
        auc_list.append(auc*100)

    print("AUC mean:{:.2f}, std:{:.4f}, max:{:.2f}, min:{:.2f}".format(
        np.mean(auc_list), np.std(auc_list), max(auc_list), min(auc_list)))


if __name__ == '__main__':

    os.makedirs(args.sample, exist_ok=True)
    main(args)