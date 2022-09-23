import sys
sys.path.append('.')

import torch.nn as nn
from network.core.net import *


# ########### Model ##########

class SynthesisModel(nn.Module):
    
    def __init__(self, latent_dim = 128) -> None:
        super().__init__()
        # channels: albedo * 3, diffuse * 3, specular * 12
        self.albedo_network = AlbedoNetwork(18, 3, image_size = 512, latent_dim = latent_dim)
        self.encoder = Encoder(512, latent_dim)
        self.decoder = Decoder(latent_dim * 2, 4, 512)

    def forward(self, x1, x2):
        texture_latent = self.encoder(x2)
        unet_recon = self.albedo_network(x1, texture_latent)
        return unet_recon, texture_latent

    def encode(self, x2):
        return self.encoder(x2)
    
    def unet(self, x1, texture_latent):
        unet_recon = self.albedo_network(x1, texture_latent)
        return unet_recon

    
    
        
        
        

