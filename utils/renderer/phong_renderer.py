import sys 
sys.path.append(".") 

import numpy as np
import torch
import cv2
import scipy
import pyshtools
import utils.renderer.shLib as shLib
import time

class torch_SH_renderer(torch.nn.Module):
	def __init__(self, level = 2, batch_size = 1, normal_size = 512, envmapshape = (16, 32), device = 'cuda'): # level is starting from 0
		super().__init__()
		self.level = level
		self.normal_size = normal_size		
		self.device = device
		self.batch_size = batch_size
		self.envmapshape = envmapshape
		self.prepare_factorial(level + level)
		self.prepare_doublefactorial(level + level)
		self.prepare_constant_factor(level)

		self.prepare_basis_val_recon(envmapshape)
	def prepare_constant_factor(self, level):
		self.constant_factor = torch.zeros((level + 1, level + 1)).to(torch.float32).to(self.device)
		for l in range(level + 1):
			for m in range(level + 1):
				self.constant_factor[l, m] = self.K(l, m)
		

	def K(self, l, m): # orthogonal bases of Fourier series 
		return (torch.sqrt(((2.0 * l + 1) * self.factorial[l - m]) / (4.0 * np.pi * self.factorial[l + m])))

	def prepare_factorial(self, n):
		self.factorial = torch.zeros(n + 1).to(torch.float32).to(self.device)
		self.factorial[0] = 1.0
		for i in range(1, n + 1):
			self.factorial[i] = self.factorial[i - 1] * i
	def prepare_doublefactorial(self, n):
		self.doublefactorial = torch.zeros(n + 1).to(torch.float32).to(self.device)
		self.doublefactorial[0] = self.doublefactorial[1] = 1.0
		for i in range(2, n + 1):
			self.doublefactorial[i] = self.doublefactorial[i - 2] * i
		
	def get_doublefactorial(self, x):
		if x < 0:
			return 1.0
		return self.doublefactorial[x]

	def P(self, l, m, x): # Legendre Polynomials
		if l == m:
			return (((-1.0) ** m) * self.get_doublefactorial(2 * m - 1) * (torch.sqrt(1 - x * x) ** m)).to(torch.float32).to(self.device)
		if l == m + 1:
			return (x * (2 * m + 1) * self.P(m, m, x)).to(torch.float32).to(self.device)
		return ((x * (2 * l - 1) * self.P(l - 1, m, x) - (l + m - 1) * self.P(l - 2, m, x)) / (l - m)).to(torch.float32).to(self.device)
	
	def gen_Y(self, l, m, theta, phi): 
		if m == 0:
			return self.constant_factor[l, 0] * self.P(l, 0, torch.cos(theta))
		if m > 0:
			return ((-1.0) ** m) * np.sqrt(2.0) * self.constant_factor[l, m] * torch.cos(phi * m) * self.P(l, m, torch.cos(theta))
		return np.sqrt(2.0) * self.constant_factor[l, -m] * torch.sin(-phi * m) * self.P(l, -m, torch.cos(theta))
	
	def gen_A(self, l):
		if l == 1:
			return 2.0 * np.pi / 3
		if l % 2 == 1:
			return 0.0
		return 2.0 * np.pi * ((-1.0) ** (l / 2 - 1)) / (l + 2) / (l - 1) * self.factorial[l] / (2.0 ** l) / (self.factorial[l // 2] ** 2)

	def xyz2uv(self, l_dir):
		# l_dir: [3, n_samples] (x, y, z)

		theta = torch.acos(-l_dir[1, :])
		phi = torch.atan2(l_dir[0, :], -l_dir[2, :])
		phi[phi < 0] += np.pi * 2
		return theta, phi


	def gen_sh(self, theta, phi, isRender = True):
		sh = torch.zeros((theta.shape[0], theta.shape[1], theta.shape[2], (self.level + 1) * (self.level + 1))).to(torch.float32).to(self.device)
		for l in range(0, self.level + 1):
			if isRender == True:
				A = self.gen_A(l)
			else:
				A = 1.0
			for m in range(-l, l + 1):	
				sh[:, :, :, l * (l + 1) + m] = self.gen_Y(l, m, theta, phi) * A
		return sh


	def forward(self, normal, sh_coeff, albedo = None):
		# sh_coeff: [bs, level * level, 3]
		# normal: [bs, 3, h, w]
		normal = torch.flip(normal, [1])
		normal = (normal - 0.5) * 2
		normal = torch.nn.functional.normalize(normal, p = 2, dim = 1)
		theta = torch.arccos(-normal[:, 1, :, :]).to(torch.float32).to(self.device)
		phi = torch.atan2(normal[:, 0, :, :], -normal[:, 2, :, :]).to(torch.float32).to(self.device)
		phi[phi < 0] += np.pi * 2
		sh = self.gen_sh(theta, phi, True).permute(0, 3, 1, 2)
		sh_coeff = sh_coeff.reshape(sh_coeff.shape[0], -1, 3)
		diffuse = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1) # axis = 1 if bz != 1
		if albedo != None:
			diffuse *= albedo
		return diffuse * 0.5

	def prepare_basis_val_recon(self, envmapshape):
		lp_samples_recon_v, lp_samples_recon_u = torch.meshgrid([torch.arange(start = 0, end = envmapshape[0], step = 1, dtype = torch.float32) / (envmapshape[0] - 1), 
									torch.arange(start = 0, end = envmapshape[1], step = 1, dtype = torch.float32) / (envmapshape[1] - 1)])

		lp_samples_recon_v = lp_samples_recon_v.flatten(start_dim = 0, end_dim = -1).to(torch.float32).to(self.device)
		lp_samples_recon_u = lp_samples_recon_u.flatten(start_dim = 0, end_dim = -1).to(torch.float32).to(self.device)

		basis_val_recon = self.gen_sh(np.pi - lp_samples_recon_v[None, None, ] * np.pi, lp_samples_recon_u[None, None, ] * np.pi * 2, False)[0, 0, ] # [num_lp_pixel, num_basis]
		
		self.basis_val_recon = torch.zeros((self.batch_size, envmapshape[0] * envmapshape[1], (self.level + 1) * (self.level + 1))).to(torch.float32).to(self.device)
		for i in range(self.batch_size):
			self.basis_val_recon[i] = basis_val_recon

	def img2shcoeff(self, lp_img, lp_recon_h = 100, lp_recon_w = 200):
		"""
		lp_img:(h,w,3)
		"""

		lp_img = torch.from_numpy(lp_img).to(torch.float32).to(self.device)
		data = scipy.io.loadmat('/data/hekai/SIPRR/utils/renderer/sphere_samples_1024.mat')
		l_dir = torch.from_numpy(data['sphere_samples']).to(torch.float32).to(self.device)
		theta, phi = self.xyz2uv(l_dir.T)
		l_samples = shLib.interpolate_bilinear(lp_img, phi[None] / np.pi / 2 * float(lp_img.shape[1] - 1), theta[None] / np.pi * float(lp_img.shape[0] - 1))[0, :]
		basis_val = self.gen_sh(np.pi - theta[None, None, ], phi[None, None, ], False)[0, 0, ] # [num_sample, num_basis]
		coeff = shLib.fit_sh_coeff(samples = l_samples, sh_basis_val = basis_val) # [num_lighting, num_basis, num_channel]
		return coeff

	def shcoeff2shimg(self, coeff, batch_size = None):
		if batch_size is None:
			batch_size = self.batch_size
		coeff = coeff.view(batch_size, (self.level + 1) ** 2, 3).to(torch.float32)
		lp_recon = shLib.reconstruct_sh(coeff, self.basis_val_recon[0:batch_size, ]).reshape((batch_size, int(self.envmapshape[0]), int(self.envmapshape[1]), -1))
		
		return lp_recon

class torch_env_renderer(torch.nn.Module):
	def __init__(self, batch_size = 1, Normal_map_shape = (512, 512), HDR_map_shape = (16, 32, 3), device = 'cpu'):
		super().__init__()
		self.Normal_map_shape = Normal_map_shape
		self.HDR_map_shape = HDR_map_shape
		self.batch_size = batch_size
		self.device = device

		size = (batch_size, Normal_map_shape[0], Normal_map_shape[1], 3)
		self.lightmap_d = torch.zeros(size).type(torch.FloatTensor).to(device)

		self.Matl = torch.zeros((batch_size, 3, HDR_map_shape[0] * HDR_map_shape[1])).type(torch.FloatTensor).to(device)
		self.coeff = torch.zeros((batch_size, HDR_map_shape[0], 1, 3)).type(torch.FloatTensor).to(device)
		# vec_v=torch.Tensor([0, 0, 1]).type(torch.FloatTensor).to(self.device)
		matl = torch.zeros(HDR_map_shape).type(torch.FloatTensor).to(device)
		for i in range(HDR_map_shape[0]):	
			for j in range(HDR_map_shape[1]):
				phi = i / HDR_map_shape[0] * np.pi
				theta = j / HDR_map_shape[1] * np.pi * 2
				matl[i][j] = torch.Tensor([np.sin(theta) * np.sin(phi), np.cos(phi), -np.cos(theta) * np.sin(phi)]).to(self.device)
				matl[i][j] /= torch.norm(matl[i][j])
		matl = torch.t(torch.reshape(matl, (-1, 3)))
		for i in range(HDR_map_shape[0]):
			phi = i / HDR_map_shape[0] * np.pi
			self.coeff[0][i][0] = torch.Tensor([np.sin(phi), np.sin(phi), np.sin(phi)]).type(torch.FloatTensor).to(self.device)
		for i in range(batch_size):
			self.Matl[i, ] = matl
			self.coeff[i, ] = self.coeff[0, ]

	def forward(self, env, normal, albedo=None):
		# normal: [batch_size, h, w, 3]

		HDR_map = torch.from_numpy(env).type(torch.FloatTensor).to(self.device)
		HDR_map = HDR_map * self.coeff
		Normal_map = normal[:, :, :, ::-1]
		Normal_map = (Normal_map - 0.5) * 2
		Normal_map = torch.from_numpy(Normal_map).type(torch.FloatTensor).to(self.device)
		# Normal_map = torch.nn.functional.normalize(Normal_map, p = 2, dim = 3)
		self.Matn=torch.reshape(Normal_map, (self.batch_size, -1, 3))

		for channel in range(3):	
			Matd = torch.bmm(self.Matn, self.Matl)
			Matd[Matd < 0] = 0
			
			Mat_hdr = torch.reshape(HDR_map[:, :, :, channel], (self.batch_size, 1, -1))
			
			Matd[:, :, :] *= Mat_hdr
			size = (self.batch_size, self.Normal_map_shape[0], self.Normal_map_shape[1])
			self.lightmap_d[:, :, :, channel] = torch.reshape(torch.sum(Matd, dim = 2), size)
		
		lightmap_d = self.lightmap_d.permute(0, 3, 1, 2)
		diffuse = lightmap_d / 50

		if albedo != None:
			diffuse *= albedo
		return diffuse

class torch_specular_renderer(torch.nn.Module):
	def __init__(self, batch_size = 1, Normal_map_shape = (512, 512), HDR_map_shape = (16, 32, 3), roughness = 64, device = 'cpu'):
		super().__init__()
		self.Normal_map_shape = Normal_map_shape
		self.HDR_map_shape = HDR_map_shape
		self.roughness = roughness
		self.batch_size = batch_size
		self.device = device

		size = (batch_size, Normal_map_shape[0], Normal_map_shape[1], 3)
		self.lightmap_d = torch.zeros(size).type(torch.FloatTensor).to(device)
		self.lightmap_s = torch.zeros(size).type(torch.FloatTensor).to(device)

		self.Matl = torch.zeros((batch_size, 3, HDR_map_shape[0] * HDR_map_shape[1])).type(torch.FloatTensor).to(device)
		self.Math = torch.zeros((batch_size, 3, HDR_map_shape[0] * HDR_map_shape[1])).type(torch.FloatTensor).to(device) 
		self.coeff = torch.zeros((batch_size, HDR_map_shape[0], 1, 3)).type(torch.FloatTensor).to(device)

		vec_v = torch.Tensor([0, 0, 1]).type(torch.FloatTensor).to(self.device)
		matl = torch.zeros(HDR_map_shape).type(torch.FloatTensor).to(device)
		math = torch.zeros(HDR_map_shape).type(torch.FloatTensor).to(device)
		for i in range(HDR_map_shape[0]):	
			for j in range(HDR_map_shape[1]):
				phi = i / HDR_map_shape[0] * np.pi
				theta = j / HDR_map_shape[1] * np.pi * 2
				matl[i][j] = torch.Tensor([np.sin(theta) * np.sin(phi), np.cos(phi), -np.cos(theta) * np.sin(phi)]).to(self.device)
				matl[i][j] /= torch.norm(matl[i][j])
				math[i][j] = (vec_v + matl[i][j]) / torch.norm(vec_v + matl[i][j])
		matl = torch.t(torch.reshape(matl, (-1, 3)))
		math = torch.t(torch.reshape(math, (-1, 3)))
		for i in range(HDR_map_shape[0]):
			phi=i / HDR_map_shape[0] * np.pi
			self.coeff[0][i][0] = torch.Tensor([np.sin(phi), np.sin(phi), np.sin(phi)]).type(torch.FloatTensor).to(self.device)
		for i in range(batch_size):
			self.Matl[i, ] = matl
			self.Math[i, ] = math
			self.coeff[i, ] = self.coeff[0, ]

	def forward(self, env, normal):
		# normal: [batch_size, h, w, 3]
		HDR_map = env * self.coeff
		Normal_map = torch.flip(normal, [3])
		Normal_map = (Normal_map - 0.5) * 2
		Normal_map = torch.nn.functional.normalize(Normal_map, p = 2, dim = 3)
		self.Matn = torch.reshape(Normal_map, (self.batch_size, -1, 3))
		Mats = torch.bmm(self.Matn, self.Math)
		Mats[Mats < 0] = 0
		Mats = (Mats / Mats.max())
		Mats = Mats ** self.roughness
		for channel in range(3):	
			Mat_hdr = torch.reshape(HDR_map[:, :, :, channel], (self.batch_size, 1, -1))
			Mat = Mats * Mat_hdr
			size = (self.batch_size, self.Normal_map_shape[0], self.Normal_map_shape[1])
			self.lightmap_s[:, :, :, channel] = torch.reshape(torch.sum(Mat, dim = 2), size)
		specular = self.lightmap_s / 60

		return specular.permute(0, 3, 1, 2)


class numpy_specular_renderer:
	def __init__(self,  Normal_map_shape = (512, 512), HDR_map_shape = (16, 32, 3), roughness = 64): # roughness must be 8, 16, 32, 64

		self.Normal_map_shape = Normal_map_shape
		self.HDR_map_shape = HDR_map_shape
		self.roughness = roughness
		
		size = (Normal_map_shape[0], Normal_map_shape[1], 3)
		self.lightmap_s = np.zeros(size).astype(np.float32)

		self.Math = np.zeros((3, HDR_map_shape[0] * HDR_map_shape[1])).astype(np.float32)
		vec_v = np.array([0, 0, 1]).astype(np.float32)
		self.coeff = np.zeros((HDR_map_shape[0], 1, 3)).astype(np.float32)
		
		math = np.zeros(HDR_map_shape).astype(np.float32)
		for i in range(HDR_map_shape[0]):	
			for j in range(HDR_map_shape[1]):
				phi = i / HDR_map_shape[0] * np.pi
				theta = j / HDR_map_shape[1] * np.pi * 2
				matl = np.array([np.sin(theta) * np.sin(phi), np.cos(phi), -np.cos(theta) * np.sin(phi)]).astype(np.float32)
				matl /= np.linalg.norm(matl)
				math[i, j] = (vec_v + matl) / np.linalg.norm(vec_v + matl)

		math = np.reshape(math, (-1, 3)).T
		for i in range(HDR_map_shape[0]):
			phi = i / HDR_map_shape[0] * np.pi
			self.coeff[i, 0] = np.array([np.sin(phi), np.sin(phi), np.sin(phi)]).astype(np.float32)
		
		self.Math = math

	def render(self, env, normal):
		# normal: [h, w, 3]
		# env: [h', w', 3]
		HDR_map = env * self.coeff
		Normal_map = normal[:, :, ::-1]
		Normal_map = (Normal_map - 0.5) * 2
		Normal_map /= np.linalg.norm(Normal_map, axis = 2, keepdims=True)
		self.Matn = np.reshape(Normal_map, (-1, 3))
		Mats = np.matmul(self.Matn, self.Math)
		Mats[Mats < 0] = 0
		Mats = (Mats / Mats.max())
		Mats = Mats ** self.roughness
		for channel in range(3):
			
			Mat_hdr = np.reshape(HDR_map[:, :, channel], (1, -1))
			Mat = Mats * Mat_hdr
			size = (self.Normal_map_shape[0], self.Normal_map_shape[1])
			self.lightmap_s[:, :, channel] = np.reshape(np.sum(Mat, axis = 1), size)
		specular = self.lightmap_s / 60

		return specular



class numpy_SH_renderer:
	def __init__(self, level = 2, resolution = 256): # level is starting from 0
		self.level = level
		self.normal_size = resolution
		self.prepare_factorial(level + level)
		self.prepare_doublefactorial(level + level)

	def prepare_constant_factor(self, level):
		self.constant_factor = np.zeros((level + 1) * (level + 1)).astype(np.float32)
		for l in range(level + 1):
			for m in range(-level, level + 1):
				self.constant_factor[l, m] = self.K(l, m)
		

	def K(self, l, m): # orthogonal bases of Fourier series 
		return (np.sqrt(((2.0 * l + 1) * self.factorial[l - m]) / (4.0 * np.pi * self.factorial[l + m])))

	def prepare_factorial(self, n):
		self.factorial = np.zeros(n + 1).astype(np.float32)
		self.factorial[0] = 1.0
		for i in range(1, n + 1):
			self.factorial[i] = self.factorial[i - 1] * i
	def prepare_doublefactorial(self, n):
		self.doublefactorial = np.zeros(n + 1).astype(np.float32)
		self.doublefactorial[0] = self.doublefactorial[1] = 1.0
		for i in range(2, n + 1):
			self.doublefactorial[i] = self.doublefactorial[i - 2] * i
		
	def get_doublefactorial(self, x):
		if x < 0:
			return 1.0
		return self.doublefactorial[x]

	def P(self, l, m, x): # Legendre Polynomials
		# if l == 1 and m == -1:
		# 	return 1.0 / 2.0 * np.sqrt(1 - x ** 2)
		if l == m:
			return ((-1.0) ** m) * self.get_doublefactorial(2 * m - 1) * (np.sqrt(1 - x * x) ** m)
		if l == m + 1:
			return x * (2 * m + 1) * self.P(m, m, x)
		return (x * (2 * l - 1) * self.P(l - 1, m, x) - (l + m - 1) * self.P(l - 2, m, x)) / (l - m)
	
	def gen_Y(self, l, m, theta, phi): 
		if m == 0:
			return self.K(l, 0) * self.P(l, 0, np.cos(theta))
		if m > 0:
			return ((-1.0) ** m) * np.sqrt(2.0) * self.K(l, m) * np.cos(phi * m) * self.P(l, m, np.cos(theta))
		return np.sqrt(2.0) * self.K(l, -m) * np.sin(-phi * m) * self.P(l, -m, np.cos(theta))
	
	def gen_A(self, l):
		if l == 1:
			return 2.0 * np.pi / 3
		if l % 2 == 1:
			return 0.0
		return 2.0 * np.pi * ((-1.0) ** (l / 2 - 1)) / (l + 2) / (l - 1) * self.factorial[l] / (2.0 ** l) / (self.factorial[l // 2] ** 2)

	def xyz2uv(self, l_dir):
		# l_dir: [3, n_samples] (x, y, z)
		theta = np.arccos(-l_dir[1, :]).astype(np.float32)
		phi = np.arctan2(l_dir[0, :], -l_dir[2, :]).astype(np.float32)
		phi[phi < 0] += np.pi * 2
		return theta, phi


	def gen_sh(self, theta, phi, isRender = True):
		sh = np.zeros((theta.shape[0], theta.shape[1], (self.level + 1) * (self.level + 1))).astype(np.float32)
		# self.P_arr = np.zeros(self.level + 1, self.level + 1
		# P 可以优化！！！
		# print("!!!", theta.shape)
		for l in range(0, self.level + 1):
			if isRender == True:
				A = self.gen_A(l)
			else:
				A = 1.0
			for m in range(-l, l + 1):	
				sh[:, :, l * (l + 1) + m] = self.gen_Y(l, m, theta, phi) * A
		
		return sh


	def render(self, normal, sh_coeff, albedo = None):
		# sh_coeff: [level * level, 3]
		# sh: [level * level, h, w]
		# constant_factor: [level * level]
		normal = normal[:, :, ::-1]
		normal = (normal - 0.5) * 2
		
		theta = np.arccos(-normal[:, :, 1])
		phi = np.arctan2(normal[:, :, 0], -normal[:, :, 2])
		phi[phi < 0] += np.pi * 2
		sh = self.gen_sh(theta, phi, True).transpose(2, 0, 1)
		diffuse = np.sum(sh_coeff[:, :, None, None] * sh[:, None, :, :], 0).transpose(1, 2, 0) # axis = 1 if bz != 1
		
		if not albedo is None:
			diffuse *= albedo
		return diffuse * 0.5

	def img2shcoeff(self, lp_img):

		data = scipy.io.loadmat('./utils/renderer/sphere_samples_1024')
		l_dir = data['sphere_samples'].astype(np.float32)
		theta, phi = self.xyz2uv(l_dir.T)
		# print(theta)
		l_samples = shLib.numpy_interpolate_bilinear(lp_img, phi[None] / np.pi / 2 * float(lp_img.shape[1] - 1), theta[None] / np.pi * float(lp_img.shape[0] - 1))[0, :]		
		basis_val = self.gen_sh(np.pi - theta[None, ], phi[None, ], False)[0] # [num_sample, num_basis]
		
		coeff = shLib.numpy_fit_sh_coeff(samples = l_samples, sh_basis_val = basis_val) # [num_lighting, num_basis, num_channel]
		return coeff

	def shcoeff2shimg(self, coeff, lp_recon_h = 100, lp_recon_w = 200):
		lmax = self.level
		lp_samples_recon_v, lp_samples_recon_u = np.meshgrid(np.arange(start = 0, stop = lp_recon_h, step = 1, dtype = np.float32) / (lp_recon_h - 1), 
									np.arange(start = 0, stop = lp_recon_w, step = 1, dtype = np.float32) / (lp_recon_w - 1))
		# lp_samples_recon_u = lp_samples_recon_u.cpu().numpy()
		# lp_samples_recon_v = lp_samples_recon_v.cpu().numpy()

		lp_samples_recon_v = lp_samples_recon_v.T.flatten()
		lp_samples_recon_u = lp_samples_recon_u.T.flatten()

		basis_val_recon = self.gen_sh(np.pi - lp_samples_recon_v[None, ] * np.pi, lp_samples_recon_u[None, ] * np.pi * 2, False)[0] # [num_lp_pixel, num_basis]
			
		coeff = np.reshape(coeff, ((lmax + 1) ** 2, 3)).astype(np.float32)
		lp_recon = shLib.numpy_reconstruct_sh(coeff, basis_val_recon).reshape((int(lp_recon_h), int(lp_recon_w), -1))
		lp_recon = lp_recon.astype('float32')
		return lp_recon


from utils.renderer.olat_renderer import np_olat_render
import glob
import os
import random

def augment_hdr(hdr):
	# white hdr generation
	gray_hdr = hdr.mean(axis=2)
	new_hdr = cv2.merge([gray_hdr for _ in range(3)])
	if random.random() > 0.85:
		hdr = new_hdr
	else:
		blend_weight = random.random()
		hdr = blend_weight * hdr + (1-blend_weight) * new_hdr

	# random rotate
	rot_pix = int(random.random() * hdr.shape[1])
	new_hdr = np.zeros_like(hdr)
	new_hdr[:, 0:rot_pix, :] = hdr[:, hdr.shape[1] - rot_pix:, :]
	new_hdr[:, rot_pix:, :] = hdr[:, 0:hdr.shape[1] - rot_pix, :]
	hdr = new_hdr

	# adjust contrast
	new_contrast = random.uniform(0.2, 1) * hdr.std()
	intensity = hdr.mean()
	hdr = (hdr - intensity) / hdr.std() * new_contrast + intensity

	hdr[hdr < 0] = 0
	return hdr
def augment_data(image, hdr, do_color=True, do_trans=True):
        if do_color:
            # color:
            # 1. gamma:
            gamma = random.uniform(1, 2.2)
            # 2. adjust color
            factor = image.max() / random.uniform(0.9, 1.2)
            image /= factor
            hdr /= factor

            if image.mean() < 0.1:
                factor = image.mean() / 0.1
                image /= factor
                hdr /= factor

        return image, hdr
	

from utils.renderer.phongshading_cuda import PhongShading
phong = PhongShading()
if __name__ == "__main__":
	# normal = cv2.imread("/data/shared/TP/hk1/normal_refine/00000.png").astype(np.float32) / 255
	# albedo = cv2.imread("/data/shared/TP/hk1/albedo_refine/00000.png").astype(np.float32) / 255
	
	posi1 = "/data/shared/TP/lights_directions_rotated_96_refine.txt"
	posi2 = '/data/hekai/512_jit/lights_directions_rotated.txt'
	
	olat_render1 = np_olat_render(posi1)
	olat_render2 = np_olat_render(posi2)
	datamen_pic_folders1 = sorted(
	    glob.glob(os.path.join("/data/shared/TP/jywq2/olat/*.png"))
	)[:96] 
	datamen_pic_folders2 = sorted(
	    glob.glob(os.path.join("/data/hekai/512_jit/image/001/OUTPUT/*.png"))
	)[:114]
	olats1 = [cv2.resize(cv2.imread(path) / 255, (512, 512)) 
		for path in datamen_pic_folders1]
	olats2 = [cv2.resize(cv2.imread(path) / 255, (512, 512)) 
		for path in datamen_pic_folders2]
	# hdr_path = sorted(
	#     glob.glob(os.path.join("/data/hekai/512_jit/label/*.hdr"))
	# )
	os.makedirs('./workspace/output/test_new_hdr/', exist_ok=True)
	# for i, path in enumerate(hdr_path):
 	# print(i)
	# normal = cv2.imread("/data/hekai/comparison/set16/normal/00001.png").astype(np.float32) / 255
	# albedo = cv2.imread("/data/hekai/512_jit/albedo_refine/001/00000.png").astype(np.float32) / 255
	# _env = cv2.imread(path, -1).astype(np.float32)
	# print(_env.max(), _env.min(), _env.mean())
	# pic2 = olat_render2.render(_env, olats2)
	# pic2, _env = augment_data(pic2, _env)
	# phong_renderer = torch_env_renderer(device = 'cuda')
	# diffuse = phong_renderer(_env[None], normal[None])[0].cpu().detach().numpy().transpose(1, 2, 0)
	
	# torch_normal = torch.from_numpy(normal).to('cuda').to(torch.float32)[None]
	# torch_env = torch.from_numpy(_env).to('cuda').to(torch.float32)[None]
	# _, ful = phong.shading(torch_env, torch_normal)
	# ful = ful[0].cpu().detach().numpy().transpose(1, 2, 0)
	
	# print(diffuse.max())
	# np_sh_renderer = numpy_SH_renderer(level = 2, resolution=512)
	# env = cv2.resize(cv2.imread("/data/hekai/512_jit/label/0000000001.hdr", -1).astype(np.float32), (200, 100))
	# print("!!!", env.min())
	# sh_coeff = np_sh_renderer.img2shcoeff(env)

	# show_recon_np = np_sh_renderer.render(normal, sh_coeff)
	
	sh_renderer = torch_SH_renderer(level = 2, batch_size = 1)
	hdr = cv2.imread('/data/hekai/512_jit/label/0000000001.hdr', -1)
	# sh_hdr = torch.from_numpy(hdr).to('cuda').to(torch.float32)
	sh_coeff = sh_renderer.img2shcoeff(hdr)
	# print(sh_coeff)
	# normal = torch.from_numpy(normal).to('cuda').to(torch.float32).permute(2, 0, 1)[None]
	# torch_albedo = torch.from_numpy(albedo).to('cuda').to(torch.float32).permute(2, 0, 1)[None]
	env = sh_renderer.shcoeff2shimg(sh_coeff)[0].cpu().numpy()
	# show_recon = sh_renderer(normal.clone(), sh_coeff[None])[0].cpu().detach().numpy().transpose(1, 2, 0) 
	# albedo *= 2.7
	# show_recon2 /= np.mean(show_recon2 * albedo) / pic3.mean()
	# diffuse2 /= np.mean(diffuse2 * albedo) / pic3.mean()
	# _env = cv2.resize(_env, (1024, 512))
	# show = np.concatenate([pic2, show_recon * albedo, diffuse * albedo, _env], axis = 1)

	# print(env.shape)
	cv2.imwrite('/data/sh_env.png', env / env.max() * 255)
	np.save('/data/sh_env.npy', env)
	# cv2.imwrite('./workspace/output/test_sh/np_render.png', show_recon_np * 255)
	# cv2.imwrite('./workspace/output/test_sh/diffuse.png', diffuse * 255)
	# cv2.imwrite('./workspace/output/test_sh/env.png', env * 255)














	
	