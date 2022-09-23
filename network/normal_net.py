import sys 
sys.path.append("./network") 

from network.znet_parts import *
from math import log2
import torch.nn as nn
import torch 

class coarse_normal_net(nn.Module):
    feature_size_log = 4
    feature_channel_log = 9

    def __init__(self, input_channels, output_channels, image_size = 512, device = 'cuda', use_skip = True, use_upconv = False):
        super().__init__()
        self.image_size = image_size
        self.image_size_log = int(log2(image_size))
        self.device = device

        setattr(self, "to_feature", ConvBlock(2**self.feature_channel_log, 2**self.feature_channel_log, norm_last=False))
        setattr(self, "feature_out", ConvBlock(2**self.feature_channel_log, 2**self.feature_channel_log))

        input_channel_log = self.feature_channel_log
        for input_size_log in range(self.feature_size_log + 1, self.image_size_log + 1):
            input_channel_log -= 1
            setattr(self, f"left_{input_size_log}", LeftBlock(2**input_channel_log))
            setattr(self, f"right_{input_size_log}", RightBlock(
                2**input_channel_log, skip=use_skip, upconv=use_upconv))

        setattr(self, f"entry_{self.image_size_log}", EntryBlock(input_channels, 2**input_channel_log))
        setattr(self, f"exit_{self.image_size_log}", ExitBlock(2**input_channel_log, output_channels))

    def forward(self, image, bound_hint=None, label_real=None, image_second=None):

        if bound_hint is not None:
            assert bound_hint.shape[1] == 1
        image_raw = image
        image = getattr(self, f"entry_{self.image_size_log}")(image)

        skips = []
        for input_size_log in range(self.image_size_log, self.feature_size_log, -1):
            image, skip = getattr(self, f"left_{input_size_log}")(image)
            skips.append(skip)

        image = self.to_feature(image)

        for input_size_log, skip in zip(range(self.feature_size_log + 1, self.image_size_log + 1), skips[::-1]):
            image = getattr(self, f"right_{input_size_log}")(image, skip)

        image = getattr(self, f"exit_{self.image_size_log}")(image)
        
        return image

    def save(self, filename):
        savefile = self._modules
        torch.save(savefile, filename)

    def load(self, filename):
        savefile = torch.load(filename, self.device)
        for name, module in savefile.items():
            try:
                getattr(self, name).load_state_dict(module.state_dict(), strict=False)
            except:
                setattr(self, name, module)
