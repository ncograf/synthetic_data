from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from c_layer import ConditionalLayer
from torch.distributions.multivariate_normal import MultivariateNormal
from type_converter import TypeConverter


class CFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        preview: int,
        n_layer: int,
        dtype: str | torch.dtype = "float32",
        scale: float = 1,
        shift: float = 0,
    ):
        """Conditional Flow Network

        Args:
            hidden_dim (int): dimension of the hidden layers needs to be even
            seq_len (int): lenght of the input sequence
            preview (int): number of previewed elements must be even
            conditional_dim (int): size of the conditional latent representation.
            n_layer (int): number of spectral layers to be used
            init_context (array_like): initial data to sample from
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
            scale (flaot): data scaling
            shift (float): data shift
        """

        nn.Module.__init__(self)

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        if not hidden_dim % 2 == 0:
            raise ValueError("Hidden dimension needs to be even.")

        self.hidden_dim = hidden_dim
        self.context_dim = hidden_dim
        self.preview = preview
        assert preview % 2 == 0
        self.n_layer = n_layer
        self.seq_len = seq_len

        self.data_scale = scale  # float
        self.data_shift = shift  # float

        self.g = ContextNet(seq_len, hidden_dim, self.context_dim)

        self.mu = MomentNet(self.context_dim, hidden_dim // 2, preview)
        self.sigma = MomentNet(self.context_dim, hidden_dim, preview)

        self.layers = nn.ModuleList(
            [
                ConditionalLayer(
                    seq_len=preview,
                    hidden_dim=hidden_dim,
                    context_dim=self.context_dim,
                    dtype=self.dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.eye = nn.Parameter(
            torch.eye(preview, dtype=self.dtype), requires_grad=False
        )
        self.zero = nn.Parameter(
            torch.zeros(preview, dtype=self.dtype), requires_grad=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for module

        This only initalizes linear layers in the network

        Args:
            module (nn.Module): Module to apply it to.
        """

        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_normal_(
                    module.weight, gain=nn.init.calculate_gain("sigmoid")
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
            "seq_len": self.seq_len,
            "preview": self.preview,
            "n_layer": self.n_layer,
            "dtype": self.dtype_str,
            "scale": self.data_scale,
            "shift": self.data_shift,
        }

        return dict_

    def set_normilizing(self, X: torch.Tensor):
        self.data_shift = np.nanmean(X)
        self.data_scale = np.nanstd(X)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Args:
            x (torch.Tensor): Nx(seq_len + preview) signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        if x.ndim == 1:
            x = x.reshape((1, -1))

        x = (x - self.data_shift) / self.data_scale

        assert x.shape[1] == self.seq_len + self.preview

        y = x[:, -self.preview :]
        x = x[:, : -self.preview]
        x = (x - self.data_shift) / self.data_scale

        x_cond = self.g(x)
        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            y, log_jac_det = layer(y, x_cond, flip=f)

            log_jac_dets.append(log_jac_det)

        # compute 'log likelyhood' of last ouput
        dist_y = MultivariateNormal(self.zero, self.eye)
        mu, sigma = self.mu(x_cond), torch.pow(self.sigma(x_cond) ** 2, 0.25)
        y_trans = (y - mu) / sigma
        log_prob_z = dist_y.log_prob(y_trans)

        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)

        return y, log_prob_z, log_jac_det_sum

    def inverse(self, z: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): Dxoutput_dim latent space variable
            x_cond (torch.Tensor): Nxhidden_dim conditional tensor

        Returns:
            torch.Tensor: return value D dimensional
        """

        for layer, f in zip(reversed(self.layers), reversed(self.flips)):
            z = layer.inverse(z, x_cond, flip=f)

        z = z * self.data_scale + self.data_shift

        return z

    def sample_single(self, z: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Sample for a single signal

        Args:
            z (torch.Tensor): random sample for one more ouput
            signal (torch.Tensor): signal to be used as condition

        Returns:
            torch.Tensor: new sample
        """
        x_cond = self.g(signal)
        sigma = torch.pow(self.sigma(x_cond) ** 2, 0.25)
        mu = self.mu(x_cond)
        z_ = z * sigma + mu
        return self.inverse(z_, x_cond)[:, 0]

    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """Sample new series from the learn distribution with
        given initial series x

        Args:
            n (int): number of series to sample
            x (torch.Tensor): Initial series to start sampling

        Returns:
            Tensor: signals in the signal space
        """
        self.eval()

        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape((1, -1))

            x = x.to(self.dtype)

            dist_z = MultivariateNormal(self.zero, self.eye)
            z = dist_z.rsample(sample_shape=(x.shape[0], n))

            signals = torch.zeros((x.shape[0], self.seq_len + n), dtype=self.dtype).to(z.device)
            signals[:, : self.seq_len] = x
            for i in range(z.shape[1]):
                curr_signal = signals[:, i : self.seq_len + i]
                signals[:, self.seq_len + i] = self.sample_single(z[:, i], curr_signal)

        return signals[:, self.seq_len :]


class ContextNet(nn.Module):
    def __init__(self, input_len: int, hidden_dim: int, output_len: int):
        nn.Module.__init__(self)
        self.layer_one = nn.Linear(input_len, hidden_dim)
        self.layer_two = nn.Linear(hidden_dim, hidden_dim)
        self.layer_three = nn.Linear(hidden_dim, hidden_dim)
        self.layer_four = nn.Linear(hidden_dim, output_len)
        self.norm_one = nn.BatchNorm1d(hidden_dim)
        self.norm_two = nn.BatchNorm1d(hidden_dim)
        self.norm_three = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.norm_one(self.layer_one(x)))
        x = torch.tanh(self.norm_two(self.layer_two(x)))
        x = torch.tanh(self.norm_three(self.layer_three(x)))
        x = self.layer_four(x)

        return x


class MomentNet(nn.Module):
    def __init__(self, input_len: int, hidden_dim: int, output_len: int):
        nn.Module.__init__(self)
        self.layer_one = nn.Linear(input_len, hidden_dim)
        self.layer_two = nn.Linear(hidden_dim, hidden_dim)
        self.layer_three = nn.Linear(hidden_dim, output_len)

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.layer_one(x))
        x = torch.tanh(self.layer_two(x))
        x = self.layer_three(x)

        return x
