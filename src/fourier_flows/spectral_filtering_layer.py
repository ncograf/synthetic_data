from typing import Tuple

import torch
import torch.nn as nn
from type_converter import TypeConverter


class SpectralFilteringLayer(nn.Module):
    def __init__(self, T: int, hidden_dim: int, dtype: str = "float64"):
        """Spectral filtering layer for seqences

        see https://arxiv.org/abs/1605.08803 for implementation details

        Args:
            D (int): Number of series
            T (int): individual series lenght
            hidden_dim (int): size of the hidden layers in neural network
        """

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        nn.Module.__init__(self)

        self.count = 0

        self.split_size = T // 2 + 1

        self.H_net = nn.Sequential(
            nn.Linear(self.split_size, hidden_dim, dtype=self.dtype),
            nn.LayerNorm(hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.LayerNorm(hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.split_size, dtype=self.dtype),
            nn.Tanh(),
        )

        self.M_net = nn.Sequential(
            nn.Linear(self.split_size, hidden_dim, dtype=self.dtype),
            nn.LayerNorm(hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.LayerNorm(hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.split_size, dtype=self.dtype),
        )

    def forward(self, x: torch.Tensor, flip: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute foward step of method proposed in
        https://arxiv.org/abs/1605.08803

        Args:
            x (torch.Tensor): Dx(T // 2 + 1)x2 input tensor
            flip (bool): Whether or not to flip the last dimensions

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: latent variable Z and det(J(f))
        """

        x_re, x_im = x[:, :, 0], x[:, :, 1]
        if flip:
            x_re, x_im = x_im, x_re

        H = self.H_net(x_re)
        H = torch.sigmoid(H)

        M = self.M_net(x_re)

        Y_1 = x_re
        Y_2 = H * x_im + M

        if flip:
            Y_2, Y_1 = Y_1, Y_2

        Y = torch.stack([Y_1, Y_2], dim=-1)

        # The jacobian is a diagonal matrix for each time series and hence
        # see https://arxiv.org/abs/1605.08803
        log_jac_det = torch.sum(torch.log(H), dim=-1)

        return Y, log_jac_det

    def inverse(self, y: torch.Tensor, flip: bool) -> torch.Tensor:
        """Computes the inverse transform of the spectral filter

        Args:
            y (torch.Tensor): Dx(T // 2 + 1)x2 input latent variable
            flip (bool): whether or not to flip the real and imag

        Returns:
            torch.Tensor: complex input to original application
        """

        y_real, y_imag = y[:, :, 0], y[:, :, 1]

        if flip:
            y_imag, y_real = y_real, y_imag

        x_real = y_real

        H = self.H_net(y_real)
        H = torch.sigmoid(H)
        M = self.M_net(y_real)

        x_imag = (y_imag - M) / H

        if flip:
            x_imag, x_real = x_real, x_imag

        x_complex = torch.stack([x_real, x_imag], dim=-1)

        return x_complex
