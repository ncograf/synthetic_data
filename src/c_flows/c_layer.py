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
        self, y: torch.Tensor, x_cond: torch.Tensor, flip: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute foward step of method proposed in
        http://arxiv.org/abs/1912.00042

        Args:
            y (torch.Tensor): batch_size x preview_len
            x_cond (torch.Tensor): batch_size x hidden_dim context tensor
            flip (bool): Whether or not to flip the last dimensions

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: latent variable y' and det(J(f))
        """

        y0, y1 = y[:, : self.seq_len // 2], y[:, self.seq_len // 2 :]
        if flip:
            y0, y1 = y1, y0

        y1_x = torch.cat([y1, x_cond], dim=1)
        H = self.pos_f(self.s(y1_x))
        M = self.t(y1_x)

        yp_0 = H * y0 + M
        yp_1 = y1

        if flip:
            yp_1, yp_0 = yp_0, yp_1

        yp = torch.cat([yp_0, yp_1], dim=-1)

        # The jacobian is a diagonal matrix for each time series and hence
        # see https://arxiv.org/abs/1605.08803
        log_jac_det = torch.sum(torch.log(H), dim=-1)

        return yp, log_jac_det

    def inverse(
        self, yp: torch.Tensor, x_cond: torch.Tensor, flip: bool
    ) -> torch.Tensor:
        """Computes the inverse transform of the spectral filter

        Args:
            yp (torch.Tensor): batch_size x preview
            x_cond (torch.Tensor): batch_size x self.hidden_dim
            flip (bool): whether or not to flip the real and imag

        Returns:
            torch.Tensor: y the inverse of y'
        """

        y0, y1 = yp[:, : self.seq_len // 2], yp[:, self.seq_len // 2 :]

        if flip:
            y1, y0 = y0, y1

        y1_x = torch.cat([y1, x_cond], dim=1)
        H = self.pos_f(self.s(y1_x))
        M = self.t(y1_x)

        y0 = (y0 - M) / H
        y1 = y1_x[:, : self.seq_len // 2]

        if flip:
            y1, y0 = y0, y1

        y = torch.cat([y0, y1], dim=-1)

        return y


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
