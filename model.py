import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class CRsAE1d(nn.Module):

    def __init__(self, T, input_size, code_size, kernel_size, kernels, channels=1):
        """Model from Constrained Recurrent Sparse Auto-Encoders paper"""
        super(CRsAE1d, self).__init__()
        self.T = T
        self.input_size = input_size
        self.code_size = code_size
        self.H = nn.Parameter(torch.randn(kernel_size, kernel_size, kernels, channels))     # self.H is the decoder, torch.transpose(self.H) is the encoder
        eigenvalues, _ = torch.eig(torch.mm(torch.t(self.H), self.H))
        max_eigenvalue = torch.max(eigenvalues)
        self.lam = nn.Parameter(torch.tensor([max_eigenvalue + 1]))        # choice of value is arbitrary, might wanna change that later
        self.L = nn.Parameter(torch.zeros(1))

    def forward(self, signal):
        """ Algorithm 1 from the paper, runs convolutional learned FISTA. The signals is denoted 'y' in the paper."""
        z = torch.zeros(self.code_size, 1)
        prev_z = torch.zeros(self.code_size, 1)
        s = 0
        prev_s = 0
        soft_threshold = nn.Softshrink(self.lam.item() / self.L.item())
        for t in range(self.T):
            s = (1+(1+4*prev_s**2)**0.5)/2      # line 3 in Algorithm 1

            w = z + (prev_s - 1)/s*(z - prev_z)     # line 4

            c = w + 1/self.L*F.conv1d(signal - F.conv1d(w, self.H), torch.transpose(self.H))        # line 5

            prev_z = z      # line 6
            z = soft_threshold(c)

        return z        # maybe return F.conv1d(z, self.H)?


class LCSC1d(nn.Module):

    def __init__(self, K, input_size, code_size, kernel_size, kernels, channels=1):
        """The model for Learned Convolutional Sparse Coding for 1d signals.
        Kernel size is 's' in the paper, kernels is 'm' the number of kernels"""
        super(LCSC1d, self).__init__()
        self.K = K
        self.input_size = input_size
        self.code_size = code_size
        self.encoder = nn.Parameter(torch.randn(kernel_size, kernel_size, channels, kernels))
        self.decoder = nn.Parameter(torch.randn(kernel_size, kernel_size, kernels, channels))
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, signal):
        """Applies convolutional LISTA on input signal. The signal is denoted 'x' in Raja's paper."""
        z = torch.zeros(self.code_size, 1)
        soft_threshold = nn.Softshrink(self.theta)
        for t in range(self.K):
            v = F.conv1d(z, self.decoder)    # v is a temporary value for computation
            v = signal - v
            v = self.encoder(v)
            v = F.conv1d(v, self.encoder)
            z = soft_threshold(z + v)

        return z        # maybe return F.conv1d(z, self.decoder)?
