import sys
from scipy.__config__ import show 
sys.path.append(".") 
import torch
import numpy as np
import cv2
from tqdm import tqdm
import os
from glob import glob

import utils.renderer.phongshading_cuda as phongshading

################## config ##################

### network
from network.relighting_net import SynthesisModel
network = SynthesisModel()

# workspace
default_input_pic_folder    = 'test_data/img'
default_input_mask_folder   = 'test_data/mask'
default_input_normal_folder = 'test_data/normal'
default_input_albedo_folder = 'test_data/albedo'
default_input_env_folder    = 'test_data/env'
default_out_folder          = 'test_data/relight'
default_cp_folder           = 'checkpoints/relight_latent'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pic',    type = str,   default = default_input_pic_folder,    help = 'Path of img input')
parser.add_argument('--mask',   type = str,   default = default_input_mask_folder,   help = 'Path of img mask')
parser.add_argument('--normal', type = str,   default = default_input_normal_folder, help = 'Path of normal input')
parser.add_argument('--albedo', type = str,   default = default_input_albedo_folder, help = 'Path of albedo input')
parser.add_argument('--env',    type = str,   default = default_input_env_folder,    help = 'Path of relight envmap')
parser.add_argument('--out',    type = str,   default = default_out_folder,          help = 'Output folder')
parser.add_argument('--cp',     type = str,   default = default_cp_folder,           help = 'Path of relight checkpoint')
parser.add_argument('--size',   type = int,   default = 512,                         help = 'Img size')
parser.add_argument('--gamma',  type = float, default = 1.7,                         help = 'Gamma of output img')

args = parser.parse_args()

input_pic_folder    = args.pic
input_mask_folder   = args.mask
input_normal_folder = args.normal
input_albedo_folder = args.albedo
input_env_folder    = args.env
out_folder          = args.out
cp_folder           = args.cp
size                = args.size
gamma               = args.gamma

os.makedirs(out_folder, exist_ok = True)
############################################


network.load_state_dict(torch.load(os.path.join(cp_folder, 'best.pth')))
network.cuda()
network.eval()

img_folder    = sorted(glob(os.path.join(input_pic_folder,    '*')))
mask_folder   = sorted(glob(os.path.join(input_mask_folder,   '*')))
normal_folder = sorted(glob(os.path.join(input_normal_folder, '*')))
albedo_folder = sorted(glob(os.path.join(input_albedo_folder, '*')))
env_folder    = sorted(glob(os.path.join(input_env_folder,    '*')))


phong_render = phongshading.PhongShading(batch_size = 1, Normal_map_shape = (size, size), device = 'cuda')

with torch.no_grad():
	for i, (img_path, mask_path, normal_path, albedo_path, env_path) in tqdm(enumerate(zip(img_folder, mask_folder, normal_folder, albedo_folder, env_folder)), total = len(img_folder)):
		img    = cv2.resize(cv2.imread(img_path), (size, size)).astype(np.float32) / 255.0
		mask   = cv2.resize(cv2.imread(mask_path), (size, size)).astype(np.float32) / 255.0
		normal = cv2.resize(cv2.imread(normal_path), (size, size)).astype(np.float32) / 255.0
		albedo = cv2.resize(cv2.imread(albedo_path), (size, size)).astype(np.float32) / 255.0
		env    = cv2.resize(cv2.imread(env_path, -1), (32, 16)).astype(np.float32)

		img = img * mask
		normal = normal * mask
		albedo = albedo * mask

		torch_img    = torch.from_numpy(img[None]).to('cuda').to(torch.float32).permute(0, 3, 1, 2)
		torch_normal = torch.from_numpy(normal[None]).to('cuda').to(torch.float32).permute(0, 3, 1, 2)
		torch_albedo = torch.from_numpy(albedo[None]).to('cuda').to(torch.float32).permute(0, 3, 1, 2)
		torch_env    = torch.from_numpy(env[None]).to('cuda').to(torch.float32)
		
		diffuse, specular = phong_render.shading(torch_env, torch_normal.clone().permute(0, 2, 3, 1))
		diffuse = diffuse * torch_albedo

		### relight network forward
		latent = network.encode(torch_img)
		cat_img = torch.cat([torch_albedo, diffuse, specular], axis = 1)

		pred_image = network.unet(cat_img, latent)


		show_image = pred_image[0].cpu().detach().numpy().transpose(1, 2, 0) * mask
		show_image = show_image ** (1.0 / gamma)
		cv2.imwrite(os.path.join(out_folder, os.path.basename(img_path)), show_image * 255)


