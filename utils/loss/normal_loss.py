import sys 
sys.path.append(".") 
import torch.nn as nn
import torch

eps = 1e-8
class normal_loss(nn.Module):
	def __init__(self):
		super(normal_loss, self).__init__()

	def criterion(self, normal_pred, normal_tgt):
		return torch.mean(1 - torch.sum(normal_pred * normal_tgt, axis = 1) / torch.norm(normal_pred + eps, p = 2, dim = 1) / torch.norm(normal_tgt + eps, p = 2, dim = 1))
	
	def forward(self, normal, ground_truth, mask = None):
		normal = (normal - 0.5) * 2
		ground_truth = (ground_truth - 0.5) * 2
		if mask is None:
			loss = self.criterion(normal, ground_truth)
		else:
			loss = self.criterion(normal * mask, ground_truth * mask)
		return loss

