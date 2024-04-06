import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions.multivariate_normal import MultivariateNormal

class SpectralFilteringLayer(nn.Module):
    
    def __init__(self, D : int, T : int, hidden_dim : int):
        """Spectral filtering layer for seqences

        see https://arxiv.org/abs/1605.08803 for implementation details

        Args:
            D (int): Number of series
            T (int): individual series lenght
            hidden_dim (int): size of the hidden layers in neural network
        """
        
        nn.Module.__init__(self)
        
        self.D = D
        self.split_size = T // 2 + 1
        
        self.H_net = nn.Sequential(
            nn.Linear(self.split_size, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.split_size),
        )

        self.M_net = nn.Sequential(
            nn.Linear(self.split_size, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.split_size),
        )
        
    def forward(self, x : torch.Tensor, flip : bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute foward step of method proposed in 
        https://arxiv.org/abs/1605.08803

        Args:
            x (torch.Tensor): Dx(T // 2 + 1)x2 input tensor
            flip (bool): Whether or not to flip the last dimensions

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: latent variable Z and det(J(f))
        """
        if flip:
            x_re = x[:,:,0]
            x_im = x[:,:,1]
        else:
            x_im = x[:,:,0]
            x_re = x[:,:,1]

        log_H = self.H_net(x_im)
        H = torch.exp(log_H)

        M = self.M_net(x_im)
        
        Y_1 = H * x_re + M
        Y_2 = x_im
        
        if flip:
            Y_2, Y_1 = Y_1, Y_2
        
        Y = torch.stack([Y_1, Y_2], dim=-1)
        
        # The jacobian is a diagonal matrix for each time series and hence 
        # see https://arxiv.org/abs/1605.08803
        log_jac_det = torch.sum(log_H, dim=-1)
        
        return Y, log_jac_det

    def inverse(self, y : torch.Tensor, flip : bool) -> torch.Tensor:
        """Computes the inverse transform of the spectral filter

        Args:
            y (torch.Tensor): Dx(T // 2 + 1)x2 input latent variable
            flip (bool): whether or not to flip the real and imag

        Returns:
            torch.Tensor: complex input to original application
        """
        
        y_real, y_imag = y[:,:,0], y[:,:,1]
        
        if flip:
            y_imag, y_real = y_real, y_imag
            
        x_imag = y_imag
        
        log_H = self.H_net(x_imag)
        H = torch.exp(log_H)
        M = self.M_net(x_imag)
        
        x_real = (y_real - M) / H
        
        if flip:
            x_imag, x_real = x_real, x_imag
        
        x_complex = torch.stack([x_real, x_imag], dim=-1)
        
        return x_complex
        