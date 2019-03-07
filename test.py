import model
import torch
import torch.functional as F
import random
import conv_dict
import scipy
from scipy.linalg import circulant
import numpy as np

# local_dict1 = circulant([1,2,3,0,0])
# local_dict2 = circulant([4,5,6,0,0])
# local_dict = np.concatenate((local_dict1, local_dict2), axis=1)
# print(local_dict)
random.seed(831997)
torch.manual_seed(831997)
# dict = conv_dict.create_dict(100, n=3, m=2)

net = model.CRsAE1d(T=10, lam=0.5, input_size=5, code_size=10, kernel_size=3, kernels=2)        # parameters matter
signal = torch.tensor([[[1, 0.5, 0, -0.5, -1]]])

net.forward(signal)