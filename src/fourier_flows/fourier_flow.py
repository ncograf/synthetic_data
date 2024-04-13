import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions.multivariate_normal import MultivariateNormal
from spectral_filtering_layer import SpectralFilteringLayer
from fourier_transform_layer import FourierTransformLayer

class FourierFlow(nn.Module):

    def __init__(self, hidden_dim : int, D : int, T : int, n_layer : int):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers
            D (int): Sample set size
            T (int): Time series size
            n_layer (int): number of spectral layers to be used
        """
        
        nn.Module.__init__(self)
        
        self.D = D
        self.T = T
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer

        self.latent_size = T // 2 + 1
        mu = torch.zeros(2 * self.latent_size)
        sigma = torch.eye(2 * self.latent_size)
        
        self.dist_z = MultivariateNormal(mu, sigma)
    
        self.layers = nn.ModuleList(
            [SpectralFilteringLayer(D=self.D, T=self.T, hidden_dim=self.hidden_dim) for _ in range(self.n_layer)]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]
        
        self.dft = FourierTransformLayer(T=self.T)
    
    
    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Args:
            x (torch.Tensor): DxT signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """
        
        x_fft : torch.Tensor = self.dft(x)
        
        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            
            x_fft, log_jac_det = layer(x_fft, flip=f)

            log_jac_dets.append(log_jac_det)
        
        # compute 'log likelyhood' of last ouput
        z = torch.cat([x_fft[:,:,0], x_fft[:,:,1]], dim=1)
        log_prob_z = self.dist_z.log_prob(z)
        
        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)
        
        return z, log_prob_z, log_jac_det_sum
    
    def inverse(self, z : torch.Tensor) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): Dx(T // 2 + 1)x2 latent space variable

        Returns:
            torch.Tensor: 
        """
        
        z_real, z_imag = z[:,:,0], z[:,:,1]
        z_complex = torch.stack([z_real, z_imag], dim=-1)

        for layer, f in zip(reversed(self.layers), reversed(self.flips)):
            
            z_complex = layer.inverse(z_complex, flip=f)

        x = self.dft.inverse(z_complex)

        return x
    
    def sample(self, n : int) -> torch.Tensor:
        """Sample new series from the learn distribution

        Args:
            n (int): number of series to sample

        Returns:
            Tensor: signals in the signal space
        """
        
        z = self.dist_z.rsample(sample_shape=(n,))
        
        z_real = z[:,:self.latent_size]
        z_imag = z[:,self.latent_size:]
        z_complex = torch.stack([z_real, z_imag], dim=-1)
        
        signals = self.inverse(z_complex)
        
        return signals