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
        preview: int,
        n_layer: int,
        dtype: str | torch.dtype = "float32",
        scale: float = 1,
        shift: float = 0,
    ):
        """Conditional Flow Network

        Args:
            hidden_dim (int): dimension of the hidden layers needs to be even
            preview (int): number of previewed elements must be even
            n_layer (int): number of spectral layers to be used
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

        self.data_scale = scale  # float
        self.data_shift = shift  # float

        self.g = ContextNet(1, hidden_dim, self.context_dim)

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
                nn.init.normal_(module.weight.data, 0, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "hidden_dim": self.hidden_dim,
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

    def inverse_step(self, z: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        assert z.shape[-1] == self.preview

        # transform z to x
        y = z
        for layer, f in zip(self.layers, self.flips):
            y = layer.inverse(y, x_cond, flip=f)
        x = y

        # compute 'log likelyhood' of last ouput
        x = x * self.data_scale + self.data_shift

        return x

    def forward_step(
        self, x: torch.Tensor, x_cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.preview

        x = (x - self.data_shift) / self.data_scale

        # transform x to z
        y = x
        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            y, log_jac_det = layer(y, x_cond, flip=f)

            log_jac_dets.append(log_jac_det)
        z = y

        # compute 'log likelyhood' of last ouput
        mu, sigma = self.mu(x_cond), torch.abs(self.sigma(x_cond))
        z = (z - mu) / sigma

        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)

        return z, log_jac_det_sum

    def f_(
        self, x: torch.Tensor, x_cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the forward pass for a seqence of time series data and start context

        Args:
            x (torch.Tensor): batch_size x seq_len, with seq_len | self.preview
            x_cond torch.Tensor: batch_size x seq_len x hidden_dim context for rnn

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: z latent variables, log_likelyhood, log_jac_det
        """
        assert x.ndim == 2
        assert x.shape[1] % self.preview == 0

        # iterate create independent steps
        x_ = x.reshape((-1, self.preview))
        x_cond_ = x_cond[:, self.preview - 1 :: self.preview, :].reshape(
            (-1, x_cond.shape[-1])
        )
        assert x_.shape[0] == x_cond_.shape[0]
        z_, log_det_jac_ = self.forward_step(x_, x_cond_)

        # compute 'log likelyhood' of last ouput
        dist_z = MultivariateNormal(self.zero, self.eye)
        log_prob_z_ = dist_z.log_prob(z_)

        z = z_.reshape(x.shape)
        log_det_jac = log_det_jac_.reshape((x.shape[0], -1))
        log_prob_z = log_prob_z_.reshape((x.shape[0], -1))

        log_det_jac = torch.mean(
            log_det_jac, dim=-1
        )  # compute average within each batch
        log_prob_z = torch.mean(log_prob_z, dim=-1)  # compute average within each batch

        return z, log_prob_z, log_det_jac

    def if_(
        self,
        z: torch.Tensor,
        x_cond: torch.Tensor,
        hc: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """_summary_

        Args:
            z (torch.Tensor): batch_size x seq_len
            x_cond (torch.Tensor): batch_size x ? x hidden_dim
            hc (Tuple[torch.Tensor, torch.Tensor]): hidden_state of last x_cond computation

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: batch_size x seq_len (x), out going hidden state
        """
        assert z.ndim == 2
        assert z.shape[1] % self.preview == 0

        assert x_cond.ndim == 3
        assert x_cond.shape[1] % self.preview == 0

        n_cond = x_cond.shape[1]
        n_seq = z.shape[1]

        # compute all outputs with available x_cond
        z_ = z[:, :n_cond].reshape((-1, self.preview))
        x_cond_ = x_cond[:, self.preview - 1 :: self.preview, :].reshape(
            (-1, x_cond.shape[-1])
        )
        assert z_.shape[0] == x_cond_.shape[0]
        x_ = self.inverse_step(z_, x_cond_)
        x_ = x_.reshape((z.shape[0], n_cond))

        # compute the remaining outputs
        x_out = []
        last_x = x_[:, -self.preview :]
        for i in range((n_seq - n_cond) // self.preview):
            start = i + n_cond
            end = i + n_cond + self.preview
            x_cond_, hc = self.g.forward(last_x.reshape(*last_x.shape, 1), hc)
            last_x = self.inverse_step(z[:, start:end], x_cond_[:, -1, :])
            x_out.append(last_x)

        # glue everything together
        x = torch.cat([x_] + x_out, dim=-1)

        return x

    def forward(
        self, x: torch.Tensor, hc=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Input x and output z distributions of x are shifted, i.e. x[:,i + 1] <-> z[:,i]

        Args:
            x (torch.Tensor): N x seq_len signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        if x.ndim == 1:
            x = x.reshape((1, -1))

        x_ = x.reshape(*x.shape, 1)
        x_cond, hc = self.g(x_, hc)

        x_cond = torch.cat([x_cond[:, :1, :], x_cond[:, :-1, :]], dim=1)

        return self.f_(x, x_cond)

    def inverse(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        hc: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Input x and output distributions of x are shifted, i.e. in_x[:,i + 1] <-> out_x[:, i]
        the previous elements are regarded as context

        Args:
            z (torch.Tensor): D x seq_len latent space variable
            x (torch.Tensor): N x ? previous seqences
            hc (Tuple[torch.Tensor, torch.Tensor] | None): context. Defaults to None

        Returns:
            torch.Tensor: return value D dimensional
        """

        x = x.reshape(*x.shape, 1)
        x_cond, hc = self.g(x, hc)

        return self.if_(z, x_cond, hc)

    def sample(self, n: int, seq_len: int, n_burn: int = 10) -> torch.Tensor:
        """Sample new series from the learn distribution with
        given initial series x

        Args:
            n (int): number of series to sample
            seq_len (int): series length to sample
            n_burn (int): number of samples to burn before sampling

        Returns:
            Tensor: signals in the signal space
        """
        assert seq_len % self.preview == 0
        self.eval()

        dist_z = MultivariateNormal(self.zero, self.eye)
        z = dist_z.rsample(sample_shape=(n, seq_len // self.preview + n_burn)).reshape(
            (n, -1)
        )
        x = torch.zeros((n, self.preview)).to(self.dtype).to(z.device)

        return self.inverse(z, x)[:, n_burn * self.preview :]


class ContextNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        nn.Module.__init__(self)

        self.rnn = nn.LSTM(input_dim, hidden_dim, 3, batch_first=True)

        self.linear_one = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hc=None):
        if hc is None:
            x, (h, c) = self.rnn(x)
        else:
            x, (h, c) = self.rnn(x, hc)

        x = torch.tanh(self.linear_one(x))
        x = self.out_layer(x)

        return x, (h, c)


class MomentNet(nn.Module):
    def __init__(self, input_len: int, hidden_dim: int, output_len: int):
        nn.Module.__init__(self)
        self.layer_one = nn.Linear(input_len, hidden_dim)
        self.layer_two = nn.Linear(hidden_dim, output_len)
        self.scale_layer = nn.Parameter(torch.ones((output_len)))

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.layer_one(x))
        x = torch.tanh(self.layer_two(x))
        x = x * self.scale_layer

        return x


if __name__ == "__main__":
    flow = CFlow(10, 2, 2)
    x = flow.sample(5, 10, 3)
    z = flow.forward(x)

    print(z)
