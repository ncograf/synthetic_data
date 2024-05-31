from typing import Tuple

import torch
import torch.nn as nn
from type_converter import TypeConverter


class ConditionalLayer(nn.Module):
    def __init__(self, output_dim, conditional_dim: int, dtype: str = "float32"):
        """Spectral filtering layer for seqences

        see http://arxiv.org/abs/1912.00042 for implementation details

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

        self.output_dim = output_dim

        if not output_dim % 2 == 0:
            raise ValueError("The output dimension must be an even number!")

        hidden_dim = output_dim // 2 + conditional_dim

        self.s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim // 2, dtype=self.dtype),
        )

        self.t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim // 2, dtype=self.dtype),
        )

        self.pos_f = torch.nn.Softplus(beta=1, threshold=30)

    def forward(
        self, z: torch.Tensor, x: torch.Tensor, flip: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute foward step of method proposed in
        http://arxiv.org/abs/1912.00042

        Args:
            x (torch.Tensor): Dx(T // 2 + 1)x2 input tensor
            flip (bool): Whether or not to flip the last dimensions

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: latent variable Z and det(J(f))
        """

        z0, z1 = z[:, : self.output_dim // 2], z[:, self.output_dim // 2 :]
        if flip:
            z0, z1 = z1, z0

        z1_x = torch.cat([z1, x], dim=1)
        H = self.pos_f(self.s(z1_x))
        M = self.t(z1_x)

        Y_0 = H * z0 + M
        Y_1 = z1

        if flip:
            Y_1, Y_0 = Y_0, Y_1

        Y = torch.cat([Y_0, Y_1], dim=-1)

        # The jacobian is a diagonal matrix for each time series and hence
        # see https://arxiv.org/abs/1605.08803
        log_jac_det = torch.sum(torch.log(H), dim=-1)

        return Y, log_jac_det

    def inverse(self, y: torch.Tensor, x: torch.Tensor, flip: bool) -> torch.Tensor:
        """Computes the inverse transform of the spectral filter

        Args:
            y (torch.Tensor): Tensor of size output dim
            x (torch.Tensor): Context tensor of size self.hidden_dim
            flip (bool): whether or not to flip the real and imag

        Returns:
            torch.Tensor: input tensor of original sequence
        """

        y0, y1 = y[:, : self.output_dim // 2], y[:, self.output_dim // 2 :]

        if flip:
            y1, y0 = y0, y1

        z1_x = torch.cat([y1, x], dim=1)
        H = self.pos_f(self.s(z1_x))
        M = self.t(z1_x)

        z0 = (y0 - M) / H
        z1 = z1_x[:, : self.output_dim // 2]

        if flip:
            z1, z0 = z0, z1

        z = torch.cat([z0, z1], dim=-1)

        return z
