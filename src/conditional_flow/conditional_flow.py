from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from conditional_fft import ConditionalFFT
from conditional_layer import ConditionalLayer
from torch.distributions.multivariate_normal import MultivariateNormal
from type_converter import TypeConverter


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_layer: int,
        dtype: str = "float32",
        dft_scale: float = 1,
        dft_shift: float = 0,
    ):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers needs to be even
            output_dim (int): dimension of the output (equals to the number of stocks to predict)
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

        if not hidden_dim % 2 == 0:
            raise ValueError("Hidden dimension needs to be even.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer

        self.g = nn.Sequential(
            nn.Linear(hidden_dim * self.output_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
        )

        self.mu = nn.Sequential(  #
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim, dtype=self.dtype),
            nn.Sigmoid(),
        )
        self.sigma = nn.Sequential(  # independent variance network
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim, dtype=self.dtype),
            nn.Softplus(beta=1, threshold=30),
        )

        self.layers = nn.ModuleList(
            [
                ConditionalLayer(
                    output_dim=output_dim, conditional_dim=hidden_dim, dtype=dtype
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.dft = ConditionalFFT(freq_filter=hidden_dim // 2)
        self.dft_scale = dft_scale  # float
        self.dft_shift = dft_shift  # float

        self.eye = nn.Parameter(
            torch.eye(output_dim, dtype=self.dtype), requires_grad=False
        )
        self.zero = nn.Parameter(
            torch.zeros(output_dim, dtype=self.dtype), requires_grad=False
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
            "output_dim": self.output_dim,
            "n_layer": self.n_layer,
            "dtype": self.dtype_str,
            "dft_scale": self.dft_scale,
            "dft_shift": self.dft_shift,
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
            x (torch.Tensor): NxTxD signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        if x.dim == 2:
            x = x.resahpe((1, *x.shape))

        if x.shape[1] < self.hidden_dim:
            raise ValueError(
                f"Context (currently {x.shape[1]}) needs to be at least as large as the hidden dimension: {self.hidden_dim}"
            )

        y = x[:, -1, :]
        x = x[:, :-1, :]
        x_fft: torch.Tensor = self.dft(x)
        x_fft = (x_fft - self.dft_shift) / self.dft_scale

        x_cond = self.g(x_fft.flatten(1, -1))
        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            y, log_jac_det = layer(y, x_cond, flip=f)

            log_jac_dets.append(log_jac_det)

        # compute 'log likelyhood' of last ouput
        dist_y = MultivariateNormal(self.zero, self.eye)
        mu, sigma = self.mu(x_cond), self.sigma(x_cond)
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

        return z

    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """Sample new series from the learn distribution with
        given initial series x

        Args:
            n (int): number of series to sample
            x (torch.Tensor): Initial series to start sampling

        Returns:
            Tensor: signals in the signal space
        """

        if x.ndim != 2 or x.shape[1] != self.output_dim:
            raise ValueError(
                "For sampling, the given x needs to be a single start seqence"
            )

        x = x.reshape((1, *x.shape))
        x = x.to(self.dtype)
        n_x = x.shape[1]

        dist_z = MultivariateNormal(self.zero, self.eye)
        z = dist_z.rsample(sample_shape=(n,))

        signals = torch.zeros((1, n_x + n, self.output_dim))
        signals[0, :n_x, :] = x
        for i in range(z.shape[0]):
            start = max(0, n_x + i - 1536)
            x_fft: torch.Tensor = self.dft(signals[0:, start : n_x + i, :])
            x_fft = (x_fft - self.dft_shift) / self.dft_scale
            x_cond = self.g(x_fft.flatten(1, -1))
            sigma = self.sigma(x_cond)
            mu = self.mu(x_cond)
            z_ = z[i] * sigma + mu
            signals[0, n_x + i, :] = self.inverse(z_, x_cond)

        return signals[0, n_x:, :]
