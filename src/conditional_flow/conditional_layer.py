from typing import Any, Dict, Tuple

import networks
import torch
import torch.nn as nn


class ConditionalLayer(nn.Module):
    def __init__(self, model_config: Dict[str, Any]):
        """Spectral filtering layer for seqences

        Args:
            model_config (Dict[str, Any]): configuration for models
                arch : model architecture
                num_model_layer : number of model layers
                hidden_dim : size of hidden layer
                dtype : torch dtype
                output_dim : output dimension (note different behaviours of MLP and RNN)
                input_dim : input dimension (note different behaviours of RNN)
                drop_out : float in [0,1) rate
                activation : (str) activation functions 'sigmoid', 'tanh', 'relu', 'softplus'
                norm : (optional) choice 'layer', 'batch', 'None' Default: 'None' only used in MLP
                reduction : (int | Literal['mean', 'sum', 'max', 'min', 'none']) reduction used for for rnn.
                bidirect : (optional) only used for rnn models. Defaults to True
        """

        nn.Module.__init__(self)

        self.output_dim = model_config["output_dim"]

        self.s = networks.model_factory(model_config)
        self.t = networks.model_factory(model_config)

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

        z0, z1 = z[:, : self.output_dim], z[:, self.output_dim :]
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

        y0, y1 = y[:, : self.output_dim], y[:, self.output_dim :]

        if flip:
            y1, y0 = y0, y1

        z1_x = torch.cat([y1, x], dim=1)
        H = self.pos_f(self.s(z1_x))
        M = self.t(z1_x)

        z0 = (y0 - M) / H
        z1 = z1_x[:, : self.output_dim]

        if flip:
            z1, z0 = z0, z1

        z = torch.cat([z0, z1], dim=-1)

        return z
