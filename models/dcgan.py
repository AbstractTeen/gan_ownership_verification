import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class Generator(nn.Module):
    def __init__(self, img_size=128, latent_dim=128):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # DCGAN for 128*128 image
        layers = []

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        layers.append(self.deconv1)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        layers.append(self.deconv2)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        layers.append(self.deconv3)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        layers.append(self.deconv4)

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        layers.append(self.deconv5)

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        layers.append(self.deconv6)

        self.all_layers = layers

    def forward(self, z):
        '''out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)'''

        feats = []
        z = z.view(-1, self.latent_dim, 1, 1)
        for deconv_layer in self.all_layers:
            z = deconv_layer(z)
            feats.append(z)
        img = z

        #return feats, img
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=128, latent_dim=128):
        super(Discriminator, self).__init__()

        layers = []
        feats = []

        #DCGAN for 128*128 image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )

        '''self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid(),
        )'''
        self.adv_layer = nn.Sequential(nn.Linear(1024*4*4, latent_dim, bias=False), nn.LeakyReLU(0.2))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 1, bias=False), nn.Sigmoid())

    def forward(self, img):
        feat = self.encoder(img)
        feat = feat.view(img.shape[0], -1)
        feat = self.adv_layer(feat)

        validity = self.classifier(feat)

        return feat, validity


# 64*64
class Generator64(nn.Module):
    # initializers
    def __init__(self, d=128, latent_dim=128):
        super(Generator64, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        input = input.view(input.shape[0], -1, 1, 1)

        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class Discriminator64(nn.Module):
    # initializers
    def __init__(self, d=128, latent_dim=128):
        super(Discriminator64, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(d*2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(d*4, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(d*8, eps=1e-04, affine=False)

        # for ogan
        self.adv_layer = nn.Sequential(nn.Linear(1024 * 4 * 4, latent_dim, bias=False), nn.LeakyReLU(0.2))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 1, bias=False), nn.Sigmoid())

    # forward method
    def forward(self, input):
        #encode
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #classify
        x = x.view(x.shape[0], -1)
        feats = self.adv_layer(x)
        validity = self.classifier(feats)

        return feats, validity

class Discriminator_oneclass64(nn.Module):
    # initializers
    def __init__(self, d=128, latent_dim=128):
        super(Discriminator_oneclass64, self).__init__()

        self.rep_dim = 64

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(d*2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(d*4, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(d*8, eps=1e-04, affine=False)

        # for ogan
        self.adv_layer = nn.Sequential(nn.Linear(1024 * 4 * 4, latent_dim, bias=False), nn.LeakyReLU(0.2))
        self.fc2 = nn.Linear(latent_dim, self.rep_dim, bias=False)

    # forward method
    def forward(self, input):
        #encode
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #classify
        x = x.view(x.shape[0], -1)
        feats = self.adv_layer(x)
        outputs = self.fc2(feats)

        return outputs

#test
"""G = Generator()
D = Discriminator()

G_in = torch.randn(1, 128).view(-1, 128, 1, 1)
D_in = torch.randn(1, 3, 128, 128)

_, G_out = G(G_in)
D_feat, D_out = D(D_in)

print("done.")"""