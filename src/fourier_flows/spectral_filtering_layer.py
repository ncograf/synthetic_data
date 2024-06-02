from typing import Literal, Tuple

import networks
import torch
import torch.nn as nn
from type_converter import TypeConverter


class SpectralFilteringLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        arch: Literal["MLP", "LSTM"],
        activation: str,
        num_model_layer: int,
        drop_out: float,
        norm: Literal["layer", "batch", "none"] = "none",
        dtype: str = "float64",
        bidirect: bool = True,
        n_stocks: int = 1,
    ):
        """Spectral filtering layer for seqences

        see https://arxiv.org/abs/1605.08803 for implementation details

        Args:
            seq_len (int): individual series lenght.
            hidden_dim (int): size of the hidden layers in neural network.
            arch (Literal['MLP', 'LSTM']): network architectures for H and M.
            activation (str): activation function to be used
            num_model_layer (int): number of layers in the model.
            drop_out (float): in [0, 1) dropout rate to be used in model.
            norm (Literal['layer', 'batch', 'none']): normalization to be used in model.
            dtype (str, optional): type used for the layer. Defaults to 'float64'
            bidirect (bool, optional): Only used for rnn. Defaults to True.
            n_stock (int, optional): Number of stocks considered only used for rnn. Defaults to 1.
        """

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        nn.Module.__init__(self)

        self.count = 0

        self.split_size = seq_len // 2 + 1

        if arch == "MLP":
            input_dim = self.split_size
            output_dim = input_dim
        else:
            input_dim = n_stocks
            output_dim = n_stocks

        config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "arch": arch,
            "activation": activation,
            "num_model_layer": num_model_layer,
            "dtype": dtype,
            "bidirect": bidirect,
            "drop_out": drop_out,
            "norm": norm,
            "n_stocks": n_stocks,
        }

        self.H_net = networks.model_factory(config)
        self.M_net = networks.model_factory(config)

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

        H_ = self.H_net(x_re)
        H = torch.nn.functional.softplus(H_)

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
        H = torch.nn.functional.softplus(H)
        M = self.M_net(y_real)

        x_imag = (y_imag - M) / H

        if flip:
            x_imag, x_real = x_real, x_imag

        x_complex = torch.stack([x_real, x_imag], dim=-1)

        return x_complex
