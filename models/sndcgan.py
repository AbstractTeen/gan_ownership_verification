import torch.nn as nn
import torch
import numpy as np

from .spectral_norm import SpectralNorm

class SNGenerator64(nn.Module):
    def __init__(self, NUM_OF_CHANNELS = 3, Z_SIZE = 128, GEN_FEATURE_MAP_SIZE = 64):
        super(SNGenerator64, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Z_SIZE, GEN_FEATURE_MAP_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 8),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*8) x 4 x 4
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 8, GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 4),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*4) x 8 x 8
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 2),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*2) x 16 x 16
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 2, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE) x 32 x 32
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE, NUM_OF_CHANNELS, 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. (NUM_OF_CHANNELS) x 64 x 64
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1, 1, 1)
        return self.main(x)


class SNDiscriminator64(nn.Module):
    def __init__(self, NUM_OF_CHANNELS = 3, Z_SIZE = 128, GEN_FEATURE_MAP_SIZE = 64):
        super(SNDiscriminator64, self).__init__()

        self.encoder = nn.Sequential(
            # input is (NUM_OF_CHANNELS) x 64 x 64
            SpectralNorm(
                nn.Conv2d(NUM_OF_CHANNELS, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE) x 32 x 32
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*2) x 16 x 16
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE * 2,GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1,bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*4) x 8 x 8
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 8, 4, 2, 1,bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*8) x 4 x 4
        )

        """self.conv_last = nn.Sequential(
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE * 8, 1, 4, 1, 0, bias=False)
            ),
            nn.Sigmoid(),
        )"""

        self.fc1 = nn.Sequential(nn.Linear(512 * 4 * 4, Z_SIZE, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.classifier = nn.Sequential(nn.Linear(Z_SIZE, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        feats = self.encoder(x).view(x.shape[0], -1) #512*4*4

        feats = self.fc1(feats)
        valid = self.classifier(feats)
        #valid = self.conv_last(feats)
        return feats, valid

class SNDiscriminator_oneclass64(nn.Module):
    def __init__(self, NUM_OF_CHANNELS = 3, Z_SIZE = 128, GEN_FEATURE_MAP_SIZE = 64):
        super(SNDiscriminator_oneclass64, self).__init__()

        self.rep_dim = 64

        self.encoder = nn.Sequential(
            # input is (NUM_OF_CHANNELS) x 64 x 64
            SpectralNorm(
                nn.Conv2d(NUM_OF_CHANNELS, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE) x 32 x 32
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*2) x 16 x 16
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE * 2,GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1,bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*4) x 8 x 8
            SpectralNorm(
                nn.Conv2d(GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 8, 4, 2, 1,bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*8) x 4 x 4
        )

        self.fc1 = nn.Sequential(nn.Linear(512 * 4 * 4, Z_SIZE, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Linear(Z_SIZE, self.rep_dim, bias=False)

    def forward(self, x):
        feats = self.encoder(x).view(x.shape[0], -1) #512*4*4
        feats = self.fc1(feats)
        outputs = self.fc2(feats)
        #valid = self.conv_last(feats)
        return outputs

"""test init"""
if __name__ == '__main__':
    g = SNGenerator64().cuda()
    d = SNDiscriminator64().cuda()

    z = torch.tensor(np.random.normal(0, 1, (32, 128)), dtype=torch.float, device="cuda")
    img = g(z)
    feats, validity = d(img)

    print("test")