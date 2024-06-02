from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn
from fourier_transform_layer import FourierTransformLayer
from spectral_filtering_layer import SpectralFilteringLayer
from torch.distributions.multivariate_normal import MultivariateNormal
from type_converter import TypeConverter


class FourierFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        num_layer: int,
        num_model_layer: int,
        drop_out: float,
        arch: Literal["MLP", "LSTM"],
        activation: Literal["sigmoid", "tanh", "softplus", "celu", "relu"],
        dtype: str = "float32",
        use_dft: bool = True,
        dft_scale: float = 1,
        dft_shift: float = 0,
        bidirect: int = True,
        norm: Literal["layer", "batch", "none"] = "none",
        n_stocks: int = 1,
    ):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers
            seq_len (int): Time series size
            num_layer (int): number of spectral layers to be used
            num_model_layer (int): number of layers in the model
            drop_out (float): rate in [0,1)
            arch (Literal['MLP', 'LSTM']): model architecture
            activation (Literal['sigmoid', 'tanh', 'relu', ...]): activation funciton to be used
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
            use_dft (bool, optional): Flag whether or not to use fourier transform.
            dft_scale (float, optional): Amount to scale dft signal. Defaults to 1.
            dft_shift (float, optional): Amount to shift dft signal. Defaults to 0.
            bidirect (bool, optional): in case of rnn model whether or not bidirectional. Defaults to True.
            norm (Literal['layer', 'batch', 'none'], optional): normalization to be used. Defaults to None.
            n_stocks (int, optional): number of stocks in model (must match input). Defaults to 1.
        """

        nn.Module.__init__(self)

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layer = num_layer
        self.arch = arch
        self.num_model_layer = num_model_layer
        self.bidirect = bidirect
        self.drop_out = drop_out
        self.norm = norm
        self.n_stocks = n_stocks
        self.activation = activation
        self.use_dft = use_dft

        self.latent_size = seq_len // 2 + 1
        self.mu = nn.Parameter(
            torch.zeros(2 * self.latent_size, dtype=self.dtype), requires_grad=False
        )
        self.sigma = nn.Parameter(
            torch.eye(2 * self.latent_size, dtype=self.dtype), requires_grad=False
        )

        self.dist_z = MultivariateNormal(self.mu, self.sigma)

        self.layers = nn.ModuleList(
            [
                SpectralFilteringLayer(
                    seq_len=self.seq_len,
                    hidden_dim=self.hidden_dim,
                    arch=arch,
                    num_model_layer=num_model_layer,
                    drop_out=drop_out,
                    dtype=self.dtype,
                    activation=self.activation,
                    bidirect=bidirect,
                    norm=norm,
                    n_stocks=n_stocks,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.dft = FourierTransformLayer(seq_len=self.seq_len, use_dft=use_dft)
        self.dft_scale = dft_scale  # float
        self.dft_shift = dft_shift  # float

        self.apply(self._init_weights)

    def _apply(self, fn):
        """Wraps the inherited apply function to reinitiate the Multivariate Normal

        Args:
            fn (function): Function to be applied
        """
        super(self.__class__, self)._apply(fn)

        # reinitialize the multivariate distribution
        self.dist_z = MultivariateNormal(self.mu, self.sigma)
        return self

    def _init_weights(self, module: nn.Module):
        """Initialize weights for module

        This only initalizes linear layers in the network

        Args:
            module (nn.Module): Module to apply it to.
        """

        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_normal_(
                    module.weight, gain=nn.init.calculate_gain("relu")
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
            "num_layer": self.n_layer,
            "num_model_layer": self.num_model_layer,
            "activation": self.activation,
            "bidirect": self.bidirect,
            "n_stocks": self.n_stocks,
            "arch": self.arch,
            "drop_out": self.drop_out,
            "norm": self.norm,
            "seq_len": self.seq_len,
            "dtype": self.dtype_str,
            "dft_shift": self.dft_shift,
            "dft_scale": self.dft_scale,
            "use_dft": self.use_dft,
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
            x (torch.Tensor): DxT signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        # if use_dft false, then this will just resize x to have the right size
        x_fft: torch.Tensor = self.dft(x)

        x_fft = (x_fft - self.dft_shift) / self.dft_scale

        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            x_fft, log_jac_det = layer(x_fft, flip=f)

            log_jac_dets.append(log_jac_det)

        # compute 'log likelyhood' of last ouput
        z = torch.cat([x_fft[:, :, 0], x_fft[:, :, 1]], dim=1)
        log_prob_z = self.dist_z.log_prob(z)

        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)

        return z, log_prob_z, log_jac_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): Dx(T // 2 + 1)x2 latent space variable

        Returns:
            torch.Tensor:
        """

        z_real, z_imag = z[:, :, 0], z[:, :, 1]
        z_complex = torch.stack([z_real, z_imag], dim=-1)

        for layer, f in zip(reversed(self.layers), reversed(self.flips)):
            z_complex = layer.inverse(z_complex, flip=f)

        z_complex = z_complex * self.dft_scale + self.dft_shift

        x = self.dft.inverse(z_complex)

        return x

    def sample(self, n: int) -> torch.Tensor:
        """Sample new series from the learn distribution

        Args:
            n (int): number of series to sample

        Returns:
            Tensor: signals in the signal space
        """

        self.eval()

        with torch.no_grad():
            z = self.dist_z.rsample(sample_shape=(n,))

            z_real = z[:, : self.latent_size]
            z_imag = z[:, self.latent_size :]
            z_complex = torch.stack([z_real, z_imag], dim=-1)

            signals = self.inverse(z_complex)

        return signals
