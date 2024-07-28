from typing import Tuple

import torch
import torch.nn as nn


class ConditionalLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        context_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        """Spectral filtering layer for seqences

        Args:
            seq_len (int): individual series lenght.
            hidden_dim (int): size of the hidden layers in neural network.
            context_dim (int): size of the context.
            dtype (torch.dtype): type used for the layer. Defaults to 'float32'
        """

        nn.Module.__init__(self)

        assert seq_len % 2 == 0

        self.seq_len = seq_len

        self.s = SNet(seq_len // 2 + context_dim, hidden_dim, seq_len // 2).to(dtype)
        self.t = SNet(seq_len // 2 + context_dim, hidden_dim, seq_len // 2).to(dtype)

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

        z0, z1 = z[:, : self.seq_len // 2], z[:, self.seq_len // 2 :]
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

        y0, y1 = y[:, : self.seq_len // 2], y[:, self.seq_len // 2 :]

        if flip:
            y1, y0 = y0, y1

        z1_x = torch.cat([y1, x], dim=1)
        H = self.pos_f(self.s(z1_x))
        M = self.t(z1_x)

        z0 = (y0 - M) / H
        z1 = z1_x[:, : self.seq_len // 2]

        if flip:
            z1, z0 = z0, z1

        z = torch.cat([z0, z1], dim=-1)

        return z


class SNet(nn.Module):
    def __init__(self, in_len: int, hidden: int, out_len: int):
        nn.Module.__init__(self)

        self.lin = nn.Linear(in_len, hidden)
        self.nmid = nn.Linear(hidden, hidden)
        self.lout = nn.Linear(hidden, out_len)

    def forward(self, x: torch.Tensor):
        # original network form
        x = torch.tanh(self.lin(x))
        x = torch.tanh(self.nmid(x))
        x = self.lout(x)

        return x
