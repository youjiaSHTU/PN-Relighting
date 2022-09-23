import sys
sys.path.append('.')
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from random import random
from torch import Tensor
from torch.autograd import Function

from network.core.op_cuda.upfirdn2d import upfirdn2d



# ########## Helper functions ##########
def fused_leaky_relu(x:Tensor, bias:Tensor=None, negative_slope:float=.2, scale:float=2**.5):
    if bias is None:
        return F.leaky_relu(x, negative_slope=0.2) * scale
    rest_dim = [1] * (x.ndim - bias.ndim - 1)
    return (F.leaky_relu( 
            x + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope)
            * scale
        )
        

# ========== Wrapped-Block classes or functions =========
class Upscale(nn.Module):
    r""" Upscale features
    """
    def __init__(self, in_ch:int, out_ch:int, kernel_size:int=3, upscale_factor:int=2) -> None:
        # sanity-check: Make sure out_ch is completely devided by upscale_facetor(block_size)
        assert out_ch % (upscale_factor*upscale_factor) == 0, f'out_ch({out_ch})/ upscale_factor^2({upscale_factor*upscale_factor}) is not zero.'
        super(Upscale, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.upscale_factor = upscale_factor
        self._build_net(in_ch, out_ch, kernel_size, upscale_factor)

    def _build_net(self, in_ch:int, out_ch:int, kernel_size:int, upscale_factor:int) -> None:
        r""" Convolute the input features and rearranges elements in a tensor of shape (N, C*r^2, H, W) -> (N, C, H*r, W*r)
             The rearrangment procedure is exactly scaling the image up.
             It seems that it is an alternative of up-sampling.
        """
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size=kernel_size)
        self.conv1_relu = nn.LeakyReLU(0.1, inplace=True)
        # Rearranges data from depth into blocks of spatial data.
        # This is the reverse transformation of SpaceToDepth.
        # More specifically, this op outputs a copy of the input tensor where values from the depth dimension (The channel dim) are moved in spatial blocks to the height and width dimensions. The attr block_size, indicates the input block_size and how the data is moved.
        # In this case, [N, C*r^2, H, W] -> [N, C, H*r, W*r]
        # This function is exactly equivalent to PixelShuffle in pytorch: 
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        self.depth_to_space = nn.PixelShuffle(upscale_factor=upscale_factor)
    def forward(self, x:Tensor) -> Tensor:
        x1 = self.conv1(x) # [N, out_ch, H, W]
        x1_relu = self.conv1_relu(x1) # [N, out_ch, H, W]
        x_scaled = self.depth_to_space(x1_relu) # [N, out_ch / 4, 2*H, 2*W]
        return x_scaled

# EqualConv2D and EqualLinear intialize and rescale weights after every updates
class EqualConv2d(nn.Module):
    """ EqualConv2D represents the equal initialized parameters using F.conv2d. 
    """
    def __init__(self, in_ch:int, out_ch:int, kernel_size:int, stride:int=1, padding:int=0, bias:bool=True) -> None:
        super().__init__()
        # Initialize kernel parameter - To make it aligned corresponding to kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        # Key scale. We rescale our conv2d layer according to the size of the kernel.
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None
        
    def forward(self, x:Tensor) -> Tensor:
        # Note that weight is rescaled after each update.
        out = F.conv2d(x, weight=self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding, dilation=1, groups=1)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}),"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLinear(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, bias:bool=True, bias_init:float=.0, lr_mul:int=1, activation:any=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim).div_(lr_mul)
            )
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1/math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
    
    def forward(self, x:Tensor) -> Tensor:
        if self.activation:
            out = F.linear(x, self.weight*self.scale)
            out = fused_leaky_relu(out, self.bias*self.lr_mul)
        else:
            out = F.linear(x, self.weight*self.scale, bias=self.bias*self.lr_mul)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class FusedLeakyReLU(nn.Module):
    """ FusedLeakyReLU = scale * LeakyReLu(x + bias)
    """
    def __init__(self, channel:int, negative_slop:float=.2, scale:float=2**.5, bias:bool=True) -> None:
        super().__init__()
        
        self.bias = nn.Parameter(
            torch.zeros(channel)
        )
        self.negative_slope = negative_slop
        self.scale = scale
    
    def forward(self, x:Tensor) -> Tensor:
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)

# ########### Encoder sturcture ##########
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k
        
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)

        return out

class ConvLayer(nn.Sequential):
    def __init__(self, in_ch:int, out_ch:int, kernel_size:int, blur_kernel:list = [1,3,3,1], down_sample=False, activate=True, bias=True) -> None:
        layers = []
        if down_sample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p+1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        
        layers.append(
            EqualConv2d(
                in_ch,
                out_ch,
                kernel_size,
                padding = self.padding,
                stride = stride,
                bias = bias and not activate
            )
        )
        if activate:
            layers.append(FusedLeakyReLU(out_ch, bias=bias))
        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, down_sample=is_downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, down_sample=is_downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Encoder(nn.Module):
    def __init__(self, size:int, latent_dim:int):
        """ Encoder for texture disentangment out of portrait.
            @ params:
                - out_ch, embedding_ch, i.e. the dimension of the latent vector.
                - kernel_size, the size of a kernel
                - reso, resolution
            @ return:
                - model 
        """
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        channels = {
            4:512, 
            8:512, 
            16:512,
            32:512, 
            64:256,
            128:128,
            256: 64,
            512:32, 
            1024:16
        }
        log_size = int(math.log(size, 2))
        convs = [ConvLayer(3, channels[size], 1)]
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i-1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        
        convs.append(EqualConv2d(in_channel, self.latent_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*convs)
    def forward(self, input):
        out = self.convs(input)
        return out.view(len(input), self.latent_dim)

# ########## Decoder ###########
class Reshape(nn.Module):
    def __init__(self, shape:tuple):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim:int=128, least_reso:int=4, reso:int=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.reso = reso
        start_index = int(math.log(least_reso, 2)) # 2: 2^2 = 4
        end_index = int(math.log(reso, 2)) # 9 : 2^9 = 512 
        channels = {
            4:512, 
            8:512, 
            16:512,
            32:512, 
            64:256,
            128:128,
            256: 64,
            512:32, 
            1024:16
        }
        ae_dim = channels[least_reso]
        # Conert latent space to img_style space
        layers = nn.ModuleList()
        layers.append(
            nn.Sequential(
                EqualLinear(latent_dim, ae_dim*least_reso**2),
                Reshape((ae_dim, least_reso, least_reso))
            )
        )
        # Upscale 
        for i in range(start_index, end_index, 1):
            in_ch = channels[2**i] 
            out_ch = channels[2**(i+1)]
            layers.append(
                nn.Sequential(
                    Upscale(in_ch, out_ch*4),
                    ResBlock(out_ch, out_ch, is_downsample=False)
                )
            )
        layers.append(
            nn.Sequential(
                ConvLayer(out_ch, 3, kernel_size=3),
                nn.Sigmoid()
            )
        )
        
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)



# #############################################################################################

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


class LeftBlock(nn.Module):
    def __init__(self, lhs_channels):
        super().__init__()
        self.conv = ConvBlock(lhs_channels, 2*lhs_channels)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.max_pool(x), x


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
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), # (W+2p-K)/s + 1 = W
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
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


class AlbedoNetwork(nn.Module):
    feature_size_log = 4
    feature_channel_log = 9

    def __init__(self, input_channels, output_channels, image_size = 512, latent_dim=128, device = 'cuda', use_skip = True, use_upconv = False):
        super().__init__()
        self.image_size = image_size
        self.image_size_log = int(math.log(image_size, 2)) # Get the image size log
        self.latent_dim = latent_dim
        self.device = device
        self.relu = nn.ReLU(True)
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.softplus = nn.Softplus()

        setattr(self, "to_feature", ConvBlock(2**self.feature_channel_log, 2**self.feature_channel_log, norm_last=False))
        setattr(self, "feature_out", ConvBlock(2**self.feature_channel_log, 2**self.feature_channel_log))

        input_channel_log = self.feature_channel_log
        for input_size_log in range(self.feature_size_log + 1, self.image_size_log + 1):
            input_channel_log -= 1
            setattr(self, f"left_{input_size_log}", LeftBlock(2**input_channel_log)) #downsample
            setattr(self, f"right_{input_size_log}", RightBlock(
                2**input_channel_log, skip=use_skip, upconv=use_upconv))
            # limzh
            setattr(self, f"latent_injection_{input_size_log}", LatentInjectionBlock(latent_dim, 2**input_channel_log, 2**input_channel_log, upsample=False))

        setattr(self, f"entry_{self.image_size_log}", EntryBlock(input_channels, 2**input_channel_log))
        setattr(self, f"exit_{self.image_size_log}", ExitBlock(2**input_channel_log, output_channels))
        

    def forward(self, image,latent, bound_hint=None, label_real=None, image_second=None):
        # image = torch.cat([image, normal], axis=1)
        if bound_hint is not None:
            assert bound_hint.shape[1] == 1
        image_raw = image
        image = getattr(self, f"entry_{self.image_size_log}")(image) # no upsample

        skips = []
        for input_size_log in range(self.image_size_log, self.feature_size_log, -1):
            image, skip = getattr(self, f"left_{input_size_log}")(image) #
            skips.append(skip)

        image = self.to_feature(image)

        for input_size_log, skip in zip(range(self.feature_size_log + 1, self.image_size_log + 1), skips[::-1]):
            image = getattr(self, f"right_{input_size_log}")(image, skip)
            # limzh
            image = getattr(self, f"latent_injection_{input_size_log}")(image, latent)

        image = getattr(self, f"exit_{self.image_size_log}")(image)
        
        return image
    


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape
        # Expand dimension of the latent
        w1 = y[:, None, :, None, None] # Nx1xchx1x1
        w2 = self.weight[None, :, :, :, :] #
        weights = w2 * (w1 + 1) # add bias, broadcast   

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x

class LatentInjectionBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters_num, upsample = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.conv1 = Conv2DMod(input_channels, filters_num, 3)
        self.to_style2 = nn.Linear(latent_dim, filters_num)
        self.conv2 = Conv2DMod(filters_num, filters_num, 3)
        self.activation = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x, istyle):
        """ x is feature map passing from the previous layer
            pre_rgb is what?
            istyle is the latent code passing in. 
        """
        if self.upsample is not None:
            x = self.upsample(x)

        style1 = self.to_style1(istyle) # Fully-connected layer converting latent to according dimensions
        x = self.conv1(x, style1) # 
        x = self.activation(x)
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)
        return x