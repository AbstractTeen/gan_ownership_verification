import torch
import numpy as np
import argparse
import os
import time
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

from data_loader import *
from utils import *
from models import Discriminator_oneclass64, SNDiscriminator_oneclass64

def main():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--img_size', type=int, default=64, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')

    parser.add_argument('--data_path', type=str, default='./data/train/imgs')
    parser.add_argument('--D_path', type=str, default='./checkpoints/D_DCGANCelebA3407.pth')
    parser.add_argument('--save_path', type=str, default='./checkpoints/d_new_best.tar')
    parser.add_argument('--result_dir', type=str, default='results')

    #training
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train.')

    parser.add_argument('--objective', default='soft-boundary', choices=['one-class', 'soft-boundary'])
    parser.add_argument('--nu', type=float, default=0.35, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
    parser.add_argument('--warm_up_n_epochs', type=int, default=30)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--lr_milestone', type=list, default=[125])

    config = parser.parse_args()

    train_dataset = MyDataset(
        config.data_path,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    test_dataset = datasets.ImageFolder(
        './data/test',
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    # initialize model
    D_new = Discriminator_oneclass64()
    D_new.cuda()

    D_state_dict = torch.load(config.D_path, map_location=lambda storage, loc: storage)
    D_new.load_state_dict(D_state_dict, strict=False)
    print("initialization from D of GAN.")

    optimizer = torch.optim.Adam(D_new.parameters(), lr=config.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestone, gamma=0.1)

    # initialize center and radius of SVDD
    center = init_center_c(config, train_loader, D_new)
    R = torch.tensor(0.0).to(config.device)
    print("initialization center c and radius R.")

    D_new.train()
    best_auc = 0.0
    auc_list = []
    loss_list = []
    for epoch in range(config.n_epochs):
        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(config.device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = D_new(inputs)
            dist = torch.sum((outputs - center) ** 2, dim=1)
            if config.objective == 'soft-boundary':
                scores = dist - R ** 2
                loss = R ** 2 + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            # Update hypersphere radius R on mini-batch distances
            if (config.objective == 'soft-boundary') and (epoch >= config.warm_up_n_epochs):
                R.data = torch.tensor(get_radius(dist, config.nu), device=config.device)

            loss_epoch += loss.item()
            n_batches += 1

        # log epoch statistics
            if idx % 20 == 0:
                epoch_train_time = time.time() - epoch_start_time
                print('  Epoch {}/{}\t Batch {} Time: {:.3f}\t Loss: {:.8f}'.format(
                    epoch + 1, config.n_epochs, idx, epoch_train_time, loss_epoch / n_batches))

        print("Testing.")

        auc = test(config, D_new, test_loader, center, R)

        auc_list.append(auc*100.0)
        loss_list.append(loss_epoch/n_batches)

        if (epoch+1) % 5 == 0:
            if auc > best_auc:
                best_auc = auc
                torch.save({'R': R, 'c': center, 'net_dict': D_new.state_dict()}, config.save_path)
                print("save best model.")

        scheduler.step()
        if epoch in config.lr_milestone:
            print('LR scheduler: new learning rate is', (scheduler.get_last_lr()))

    torch.save({'Loss': loss_list, 'Auc': auc_list}, "./checkpoints/loss_norandom.tar")
    print("Training finished. ")

if __name__ == '__main__':
    main()