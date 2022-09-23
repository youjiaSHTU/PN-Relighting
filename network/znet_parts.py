import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_last=True, ksize=3):
        super().__init__()
        if norm_last:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                # nn.ReLU(True)
            )

    def forward(self, x):
        return self.conv(x)

class ConvBlock_test(nn.Module):
    def __init__(self, in_channels, out_channels, norm_last=True, ksize=3):
        super().__init__()
        if norm_last:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize//2, stride=2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize//2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize//2),
            )

    def forward(self, x):
        return self.conv(x)

class LeftBlock(nn.Module):
    def __init__(self, lhs_channels):
        super().__init__()
        self.conv = ConvBlock(lhs_channels, 2*lhs_channels)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x) -> "done,skip":
        x = self.conv(x)
        return self.max_pool(x), x

class LeftBlock_no_maxpool(nn.Module):
    def __init__(self, lhs_channels):
        super().__init__()
        self.conv = ConvBlock(lhs_channels, 2 * lhs_channels)
        self.conv2 = ConvBlock_test(2 * lhs_channels, 2 * lhs_channels)

    def forward(self, x) -> "done,skip":
        x = self.conv(x)
        return self.conv2(x), x


class RightBlock(nn.Module):
    def __init__(self, rhs_channels, skip=True, upconv=False):
        super().__init__()
        self.skip = skip
        if skip:
            self.conv = ConvBlock(4*rhs_channels, rhs_channels)
        else:
            self.conv = ConvBlock(2*rhs_channels, rhs_channels)
        self.upconv = upconv
        if upconv:
            self.up = nn.ConvTranspose2d(2*rhs_channels, 2*rhs_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip):
        if self.upconv:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.skip:
            x = torch.cat((x, skip), 1)
        x = self.conv(x)
        return x


class EntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        return x


class ExitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()
            # nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class PixelBlock(nn.Module):
    def __init__(self, lhs_channels, pool_window_size):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(lhs_channels, 2*lhs_channels, ksize=7),
            ConvBlock(2*lhs_channels, 2*lhs_channels, ksize=7)
        )
        self.max_pool = nn.MaxPool2d(pool_window_size, stride=1, padding=pool_window_size//2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        return x
