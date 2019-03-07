import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
from scipy.linalg import circulant


class CRsAE1d(nn.Module):

    def __init__(self, T, lam, input_size, code_size, kernel_size, kernels):
        """Model from Constrained Recurrent Sparse Auto-Encoders paper"""
        super(CRsAE1d, self).__init__()
        self.T = T
        self.input_size = input_size
        self.code_size = code_size
        self.kernel_size = kernel_size
        self.kernels = kernels
        self.local_dictionary = nn.Parameter(torch.randn(kernels, kernel_size))    # self.H is the decoder, torch.transpose(self.H) is the encoder   #TODO: maybe 2d convolution?
        filters = self.local_dictionary.tolist()
        self.H = torch.tensor(np.concatenate([circulant(filter + [0]*(self.input_size - self.kernel_size)) for filter in filters], axis=1))
        print(self.H)
        print("H's shape on creation: {}".format(self.H.shape))
        eigenvalues, _ = torch.eig(torch.mm(torch.t(self.H), self.H))
        max_eigenvalue = torch.max(eigenvalues)
        self.L = max_eigenvalue.item() + 1       # choice of value is arbitrary, might wanna change that later
        self.lam = lam

    def forward(self, signal):
        """ Algorithm 1 from the paper, runs convolutional learned FISTA. The signals is denoted 'y' in the paper."""
        torch.set_default_tensor_type(torch.DoubleTensor)
        x = torch.zeros(self.code_size)
        prev_x = torch.zeros(self.code_size)
        s = 0
        prev_s = 0
        pad = self.kernel_size//2   #TODO: figure padding
        soft_threshold = nn.Softshrink(self.lam / self.L)
        for t in range(self.T):
            s = (1+(1+4*prev_s**2)**0.5)/2      # line 3 in Algorithm 1

            w = x + ((prev_s - 1)/s)*(x - prev_x)     # line 4

            # line 5
            print("H's shape: {}, w's shape: {}".format(self.H.shape, w.shape))
            v = torch.mm(self.H, w)
            print("v's shape: {}, signal's shape: {}, x: {}, local dictionary's shape: {}".format(v.shape, signal.shape, x.shape, self.local_dictionary.shape))
            v = signal - v
            c = w + (1/self.L)*F.conv_transpose1d(v, self.local_dictionary, padding=pad)       # TODO: check translation

            prev_x = x      # line 6
            x = soft_threshold(c)

        return x        # maybe return F.conv1d(z, self.H)?


class LCSC1d(nn.Module):

    def __init__(self, K, input_size, code_size, kernel_size, kernels, channels=1):
        """The model for Learned Convolutional Sparse Coding for 1d signals.
        Kernel size is 's' in the paper, kernels is 'm' the number of kernels"""
        super(LCSC1d, self).__init__()
        self.K = K
        self.input_size = input_size
        self.code_size = code_size
        self.kernel_size = kernel_size
        self.kernels = kernels
        self.channels = channels
        self.encoder = nn.Parameter(torch.randn(channels, kernels, kernel_size))
        self.decoder = nn.Parameter(torch.randn(kernels, channels, kernel_size))
        self.theta = nn.Parameter(torch.zeros(1))
        if input_size >= code_size:
            print("Code size must be larger than input size!")
            exit()
        # TODO: weird, maybe shouldn't happen for circulant matrix multiplication?
        if input_size < code_size - kernel_size + 1:
            print("Input size too small for code size!")
            exit()

    def forward(self, signal):
        """Applies convolutional LISTA on input signal. The signal is denoted 'x' in Raja's paper."""
        z = torch.zeros(1, 1, self.code_size)
        soft_threshold = nn.Softshrink(0.5)         # TODO: problem, lambda has to be a number and not a parameter!
        pad = math.ceil((self.kernel_size + self.code_size - self.input_size - 1)/2)
        for t in range(self.K):
            v = F.conv1d(z, self.decoder)    # v is a temporary value for computation
            v = signal - v
            v = F.conv_transpose1d(v, self.encoder, padding=pad)
            z = soft_threshold(z + v)
        print(z)
        return z        # maybe return F.conv1d(z, self.decoder)?
