import numpy as np
import torch
from torch.autograd import Variable

PI = torch.FloatTensor([3.1415926])

def normal_log_density(x, mean, log_std, std):
	var = std.pow(2)
	log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
	return log_density.sum(1)

