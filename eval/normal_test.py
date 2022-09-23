import sys 
sys.path.append(".") 
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import os

################## config ##################

### network
from network.normal_net import coarse_normal_net
network = coarse_normal_net(3, 3, 512)

### workspace
default_input_pic_folder    = 'test_data/img'
default_input_mask_folder   = 'test_data/mask'
default_out_folder          = 'test_data/normal'
default_cp_folder           = 'checkpoints/normal'

import argparse
parser = argparse.ArgumentParser(description='path')
parser.add_argument('--pic',    type = str,   default = default_input_pic_folder,    help = 'Path of img input')
parser.add_argument('--mask',   type = str,   default = default_input_mask_folder,   help = 'Path of img mask')
parser.add_argument('--out',    type = str,   default = default_out_folder,          help = 'Output folder')
parser.add_argument('--cp',     type = str,   default = default_cp_folder,           help = 'Path of relight checkpoint')
parser.add_argument('--size',   type = int,   default = 512,                         help = 'Img size')

args = parser.parse_args()

input_pic_folder  = args.pic
input_mask_folder = args.mask
out_folder        = args.out
cp_folder         = args.cp
size              = args.size

os.makedirs(out_folder, exist_ok = True)
os.makedirs(cp_folder, exist_ok = True)
############################################

network = torch.load(os.path.join(cp_folder, 'best.pth'))
network.cuda()
network.eval()

img_folder  = sorted(glob(os.path.join(input_pic_folder, '*')))
mask_folder = sorted(glob(os.path.join(input_mask_folder, '*')))

with torch.no_grad():
	assert (len(img_folder) == len(mask_folder)), 'Error in number of images'
	for i, (img_path, mask_path) in tqdm(enumerate(zip(img_folder, mask_folder)), total = len(img_folder)):

		img  = cv2.resize(cv2.imread(img_path), (size, size)).astype(np.float32) / 255.0
		mask = cv2.resize(cv2.imread(mask_path), (size, size)).astype(np.float32) / 255.0
		img = img * mask

		torch_img   = torch.from_numpy(img[None]).to('cuda').permute(0, 3, 1, 2)
		pred_normal = network(torch_img)
		
		np_pred_normal = pred_normal[0].cpu().numpy().transpose(1, 2, 0)
		np_pred_normal = np_pred_normal * mask
		cv2.imwrite(os.path.join(out_folder, os.path.basename(img_path)), np_pred_normal * 255)
