import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.backends import cudnn
from torch.utils import data

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        #torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        #torch.nn.init.constant_(m.bias.data, 0.0)
        pass

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def pearson(x, y):
    x_mu = torch.mean(x, dim=1, keepdim=True)
    y_mu = torch.mean(y, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    y_std = torch.std(y, dim=1, keepdim=True)
    a = torch.mean((x - x_mu) * (y - y_mu), dim=1, keepdim=True)
    b = x_std * y_std
    return a / b

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def init_center_c(config, train_loader, net, eps=0.2):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(config.output_dim, device=config.device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(config.device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


"""def compare(imgs1,imgs2):
    ssim, psnr = 0.0, 0.0
    n_sample = 0
    for i in range(imgs1.shape[0]):
        img1, img2 = imgs1[i], imgs2[i]

        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        img1_np = np.transpose(img1_np, (1, 2, 0))
        img2_np = np.transpose(img2_np, (1, 2, 0))

        tmp = structural_similarity(img1_np, img2_np, multichannel=True)

        ssim += tmp
        psnr += peak_signal_noise_ratio(img1_np, img2_np)
        n_sample += 1

    return ssim/n_sample, psnr/n_sample"""

def test(args, net, test_loader, c, R):
    idx_label_score = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to("cuda")
            outputs = net(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            if args.objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(
                labels.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist()))

    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    test_auc = roc_auc_score(labels, scores)


    ap = average_precision_score(labels, scores, average='macro', pos_label=1, sample_weight=None)
    print('AP:', ap)

    print('Test set AUC: {:.2f}%'.format(100. * test_auc))



    return test_auc

class MyDataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform=None):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.dataset = []
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        #dataset = self.dataset
        #filename, label = dataset[index]
        filename = self.images[index]
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        label = 0

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return len(self.images)