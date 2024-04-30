from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from fourier_transform_layer import FourierTransformLayer
from spectral_filtering_layer import SpectralFilteringLayer
from torch.distributions.multivariate_normal import MultivariateNormal
from type_converter import TypeConverter


class FourierFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        T: int,
        n_layer: int,
        dtype: str = "float64",
        dft_scale: float = 1,
        dft_shift: float = 0,
    ):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers
            T (int): Time series size
            n_layer (int): number of spectral layers to be used
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
            dft_scale (float, optional): Amount to scale dft signal. Defaults to 1.
            dft_shift (float, optional): Amount to shift dft signal. Defaults to 0.
        """

        nn.Module.__init__(self)

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        self.T = T
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer

        self.latent_size = T // 2 + 1
        mu = torch.zeros(2 * self.latent_size, dtype=self.dtype)
        sigma = torch.eye(2 * self.latent_size, dtype=self.dtype)

        self.dist_z = MultivariateNormal(mu, sigma)

        self.layers = nn.ModuleList(
            [
                SpectralFilteringLayer(
                    T=self.T, hidden_dim=self.hidden_dim, dtype=dtype
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.dft = FourierTransformLayer(T=self.T)
        self.dft_scale = dft_scale
        self.dft_shift = dft_shift

        self.apply(self._init_weights)
    
    def to(self, device : torch.device | str):
        """Moves the model to the device

        Args:
            device (torch.device | str): Device to move model on.
        """
        super(self.__class__, self).to(device)
        mu = torch.zeros(2 * self.latent_size, dtype=self.dtype, device=device)
        sigma = torch.eye(2 * self.latent_size, dtype=self.dtype, device=device)

        self.dist_z = MultivariateNormal(mu, sigma)


    def _init_weights(self, module: nn.Module):
        """Initialize weights for module

        This only initalizes linear layers in the network

        Args:
            module (nn.Module): Module to apply it to.
        """

        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_normal_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "hidden_dim": self.hidden_dim,
            "n_layer": self.n_layer,
            "T": self.T,
            "dtype": self.dtype_str,
            "dft_shift": self.dft_shift,
            "dft_scale": self.dft_scale,
        }

        return dict_

    def set_normilizing(self, X: torch.Tensor):
        x_fft = self.dft(X)
        self.dft_shift = torch.mean(x_fft)
        self.dft_scale = torch.std(x_fft)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Args:
            x (torch.Tensor): DxT signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        x_fft: torch.Tensor = self.dft(x)

        x_fft = (x_fft - self.dft_shift) / self.dft_scale

        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            x_fft, log_jac_det = layer(x_fft, flip=f)

            log_jac_dets.append(log_jac_det)

        # compute 'log likelyhood' of last ouput
        z = torch.cat([x_fft[:, :, 0], x_fft[:, :, 1]], dim=1)
        log_prob_z = self.dist_z.log_prob(z)

        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)

        return z, log_prob_z, log_jac_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): Dx(T // 2 + 1)x2 latent space variable

        Returns:
            torch.Tensor:
        """

        z_real, z_imag = z[:, :, 0], z[:, :, 1]
        z_complex = torch.stack([z_real, z_imag], dim=-1)

        for layer, f in zip(reversed(self.layers), reversed(self.flips)):
            z_complex = layer.inverse(z_complex, flip=f)

        z_complex = z_complex * self.dft_scale + self.dft_shift

        x = self.dft.inverse(z_complex)

        return x

    def sample(self, n: int) -> torch.Tensor:
        """Sample new series from the learn distribution

        Args:
            n (int): number of series to sample

        Returns:
            Tensor: signals in the signal space
        """

        z = self.dist_z.rsample(sample_shape=(n,))

        z_real = z[:, : self.latent_size]
        z_imag = z[:, self.latent_size :]
        z_complex = torch.stack([z_real, z_imag], dim=-1)

        signals = self.inverse(z_complex)

        return signals
