import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from torchvision import transforms
from torch.autograd import Variable

from math import exp

# SSIM Loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2
    C1 = 0.05**2
    C2 = 0.08**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, width, _) = img1.size()
        result, count = 0, 0
        # kernel = create_window(5, 3)
        kernel = torch.ones((channel, 1, 2, 2)) / 4
        if img1.is_cuda:
            kernel = kernel.cuda(img1.get_device())
        kernel = kernel.type_as(img1)
        while width >= 16:
            result = result + self.single_ssim(img1, img2)
            img1 = F.conv2d(img1, kernel, stride=2, groups=channel)
            img2 = F.conv2d(img2, kernel, stride=2, groups=channel)
            (_, channel, width, _) = img1.size()
            count += 1
        return result / count

    def single_ssim(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        self.window = window
        self.channel = channel

        return (1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2

class G_Cri(nn.Module):
    def __init__(self, l1_weight=1, l2_weight=0.1, ssim_weight = 0.1):
        super(G_Cri, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight
        print(f'[Loss weight] L1={l1_weight}, L2={l2_weight}, ssim={ssim_weight}')
        self.ssim = SSIM()
        

    def forward(self, targets, outputs):
        gen_loss_L1 = torch.mean(torch.abs(targets - outputs))
        gen_loss_L2 = torch.mean(torch.flatten(targets - outputs)**2)
        gen_loss_ssim = self.ssim(targets, outputs)
        gen_loss = gen_loss_L1 * self.l1_weight  + gen_loss_ssim * self.ssim_weight + gen_loss_L2 * self.l2_weight
        # gen_loss = gen_loss_L1 * self.l1_weight  + gen_loss_L2 * self.l2_weight 

        return gen_loss