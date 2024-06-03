from typing import Any, Dict, Literal, Tuple

import networks
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
        dim: int,
        conditional_dim: int,
        n_layer: int,
        num_model_layer: int,
        drop_out: float,
        activation: str,
        norm: Literal["layer", "batch", "none"],
        dtype: str = "float32",
        dft_scale: float = 1,
        dft_shift: float = 0,
    ):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers needs to be even
            dim (int): dimension of the output / input (equals to the number of stocks to predict)
            conditional_dim (int): size of the conditional latent representation.
            n_layer (int): number of spectral layers to be used
            num_model_layer(int): number of model layer
            drop_out (float): dropout rate in [0, 1).
            activation (str): string indicationg the activation function.
            norm (Literal['layer', 'batch', 'none']): normalization layer to be used.
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
        self.dim = dim
        self.conditional_dim = conditional_dim
        self.num_model_layer = num_model_layer
        self.n_layer = n_layer
        self.drop_out = drop_out
        self.norm = norm
        self.activation = activation

        self.dft = ConditionalFFT()
        self.dft_scale = dft_scale  # float
        self.dft_shift = dft_shift  # float

        model_g_config = {
            "arch": "LSTM",
            "num_model_layer": num_model_layer,
            "dtype": self.dtype,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.dim,
            "output_dim": self.conditional_dim,
            "activation": self.activation,
            "drop_out": self.drop_out,
            "norm": self.norm,
            "bidirect": False,
            "reduction": -1,
        }

        model_dist_config = {
            "arch": "MLP",
            "num_model_layer": num_model_layer,
            "dtype": self.dtype,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.conditional_dim,
            "activation": self.activation,
            "output_dim": self.dim,
            "drop_out": self.drop_out,
            "norm": self.norm,
        }

        model_config = {
            "arch": "MLP",
            "num_model_layer": num_model_layer,
            "dtype": self.dtype,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.dim // 2 + self.conditional_dim,
            "activation": self.activation,
            "output_dim": self.dim // 2,
            "drop_out": self.drop_out,
            "norm": self.norm,
        }

        self.g = networks.model_factory(model_g_config)

        self.mu = networks.model_factory(model_dist_config)
        self.sigma = networks.model_factory(model_dist_config)

        self.layers = nn.ModuleList(
            [
                ConditionalLayer(
                    model_config=model_config,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.eye = nn.Parameter(torch.eye(dim, dtype=self.dtype), requires_grad=False)
        self.zero = nn.Parameter(
            torch.zeros(dim, dtype=self.dtype), requires_grad=False
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
            "dim": self.dim,
            "conditional_dim": self.conditional_dim,
            "n_layer": self.n_layer,
            "num_model_layer": self.num_model_layer,
            "drop_out": self.drop_out,
            "activation": self.activation,
            "norm": self.norm,
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

        if x.ndim == 2:
            x = x.reshape((1, *x.shape))

        y = x[:, -1, :]
        x = x[:, :-1, :]
        x_fft: torch.Tensor = self.dft(x)
        x_fft = (x_fft - self.dft_shift) / self.dft_scale

        x_cond = self.g(x_fft)
        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            y, log_jac_det = layer(y, x_cond, flip=f)

            log_jac_dets.append(log_jac_det)

        # compute 'log likelyhood' of last ouput
        dist_y = MultivariateNormal(self.zero, self.eye)
        mu, sigma = self.mu(x_cond), self.sigma(x_cond) ** 2
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

    def sample_single(self, z: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Sample for a single signal

        Args:
            z (torch.Tensor): random sample for one more ouput
            signal (torch.Tensor): signal to be used as condition

        Returns:
            torch.Tensor: new sample
        """
        x_fft: torch.Tensor = self.dft(signal)
        x_fft = (x_fft - self.dft_shift) / self.dft_scale
        x_cond = self.g(x_fft)
        sigma = self.sigma(x_cond) ** 2
        mu = self.mu(x_cond)
        z_ = z * sigma + mu
        return self.inverse(z_, x_cond)

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
            if x.ndim != 2 or x.shape[1] != self.dim:
                raise ValueError(
                    "For sampling, the given x needs to be a single start seqence"
                )

            x = x.reshape((1, *x.shape))
            x = x.to(self.dtype)
            n_x = x.shape[1]

            dist_z = MultivariateNormal(self.zero, self.eye)
            z = dist_z.rsample(sample_shape=(n,))

            signals = torch.zeros((1, n_x + n, self.dim), dtype=self.dtype)
            signals[0, :n_x, :] = x
            for i in range(z.shape[0]):
                start = max(0, n_x + i - 1536)
                curr_signal = signals[0:, start : n_x + i, :]
                signals[0, n_x + i, :] = self.sample_single(z[i], curr_signal)

        return signals[0, n_x:, :]
