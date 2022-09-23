import sys 
sys.path.append(".") 
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from math import *

class pyramid_loss(nn.Module):
	def __init__(self, level = 5):
		super(pyramid_loss, self).__init__()
		self.level = level
		self.kernel = [None]
		self.prepare_GaussianKernel()

	def prepare_GaussianKernel(self):
		for k in range(1, self.level + 1):
			size = 2 * k + 1
			sum = 0.0
			self.kernel.append(torch.zeros((size, size)))
			# sigmaX = ((size - 1) * 0.5 - 1) * 0.3 + 0.8
			sigmaX = size
			for i in range(0, size):
				for j in range(0, size):
					self.kernel[k][i, j] = exp(-((i - k) * (i - k) + (j - k) * (j - k)) / (2.0 * sigmaX * sigmaX))
					sum += self.kernel[k][i, j]
			self.kernel[k] /= sum
			self.kernel[k] = self.kernel[k].to('cuda').to(torch.float32)[None][None]
			self.kernel[k] = nn.Parameter(data = self.kernel[k])
		
	def GaussianBlur(self, image, kernel_size):
		channel = image.shape[1]
		kernel = self.kernel[(kernel_size - 1) // 2]
		filter = kernel.expand(channel, 1, kernel.shape[2], kernel.shape[3])
		image = F.conv2d(image, filter, padding = (kernel_size - 1) // 2, groups = channel)
		return image

	def construct_lap_pyramid(self, image, level = 4):
		img_pyramid = torch.zeros((level, *image.shape)).to('cuda').to(torch.float32)
		img_pyramid[0] = image
		for i in range(1, level):
			img_pyramid[i] = self.GaussianBlur(image.clone(), 2 * i + 1)
		
		lap_pyramid = torch.zeros_like(img_pyramid).to('cuda').to(torch.float32)
		lap_pyramid[0] = img_pyramid[level - 1]
		for i in range(level - 1):
			lap_pyramid[i + 1] = img_pyramid[level - 2 - i] - img_pyramid[level - 1 - i]
		
		
		return lap_pyramid

	def forward(self, gt_normal, normal_pyramid, mask):
		gt_pyramid = self.construct_lap_pyramid(gt_normal)
		# print(gt_pyramid[0].shape, normal_pyramid[0].shape)
		criterion_L2 = torch.nn.MSELoss()

		return 1.0 * criterion_L2(gt_pyramid[0] * mask, normal_pyramid[0] * mask) + \
		       5.0 * criterion_L2(gt_pyramid[1] * mask, normal_pyramid[1] * mask) + \
		       10.0 * criterion_L2(gt_pyramid[2] * mask, normal_pyramid[2] * mask) + \
		       20.0 * criterion_L2(gt_pyramid[3] * mask, normal_pyramid[3] * mask)


if __name__ == "__main__":
	normal = cv2.imread('/data/hekai/512_jit/normal/001/000.png').astype(np.float32) / 255

	normal = torch.from_numpy(normal).to('cuda').to(torch.float32).permute(2, 0, 1)[None]

	l = 9
	Pyramid_loss = pyramid_loss(l)

	lap_pyramid = Pyramid_loss.construct_lap_pyramid(normal, l)
	 
	y = np.zeros((512, 512, 3)).astype(np.float32)

	for i in range(l):
		x = lap_pyramid[i]
		x = x[0].cpu().detach().numpy().transpose(1, 2, 0)
		y += x
		cv2.imwrite('./workspace/normal_pyramid/' + str(i).zfill(5) + '.png', x * 255)

	cv2.imwrite('./workspace/normal_pyramid/' + str(5).zfill(5) + '.png', y * 255)

