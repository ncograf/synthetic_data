from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from fourier_transform_layer import FourierTransformLayer
from spectral_filtering_layer import SpectralFilteringLayer
from type_converter import TypeConverter


class FourierFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        num_layer: int,
        dist_config: Dict[str, Any],
        dtype: str | torch.dtype = "float32",
        use_dft: bool = True,
    ):
        """Fourier Flow network for one dimensional time series

        Args:
            hidden_dim (int): dimension of the hidden layers
            seq_len (int): Time series size
            num_layer (int): number of spectral layers to be used
            dist_config (Dict): distribution configuration
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
            use_dft (bool, optional): Flag whether or not to use fourier transform.
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

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layer = num_layer
        self.use_dft = use_dft
        self.dist_config = dist_config

        self.loc = nn.Parameter(
            torch.ones(1) * self.dist_config["loc"], requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.ones(1) * self.dist_config["scale"], requires_grad=False
        )
        if "studentt" == self.dist_config["dist"]:
            self.df = nn.Parameter(
                torch.ones(1) * self.dist_config["df"], requires_grad=False
            )

        self.latent_size = seq_len // 2 + 1

        self.layers = nn.ModuleList(
            [
                SpectralFilteringLayer(
                    seq_len=self.seq_len,
                    hidden_dim=self.hidden_dim,
                    dtype=self.dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.flips = [True if i % 2 else False for i in range(self.n_layer)]

        self.dft = FourierTransformLayer(seq_len=self.seq_len, use_dft=use_dft)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for module

        This only initalizes linear layers in the network

        Args:
            module (nn.Module): Module to apply it to.
        """

        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.normal_(module.weight.data, 0, 0.00001)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "hidden_dim": self.hidden_dim,
            "num_layer": self.n_layer,
            "seq_len": self.seq_len,
            "dist_config": self.dist_config,
            "dtype": self.dtype_str,
            "use_dft": self.use_dft,
        }

        return dict_

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Args:
            x (torch.Tensor): DxT signal batch tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        match self.dist_config["dist"]:
            case "studentt":
                dist = torch.distributions.StudentT(self.df, self.loc, self.scale)
            case "normal":
                dist = torch.distributions.Normal(self.loc, self.scale)
            case "cauchy":
                dist = torch.distributions.Cauchy(self.loc, self.scale)
            case "laplace":
                dist = torch.distributions.Laplace(self.loc, self.scale)

        # if use_dft false, then this will just resize x to have the right size
        x_fft: torch.Tensor = self.dft(x)

        log_jac_dets = []
        for layer, f in zip(self.layers, self.flips):
            x_fft, log_jac_det = layer(x_fft, flip=f)

            log_jac_dets.append(log_jac_det)

        # inverse dft
        z = self.dft.inverse(x_fft)

        # compute 'log likelyhood' of last ouput
        d = self.seq_len * self.n_layer
        z = torch.nan_to_num(z, 0, 0, 0)  # gradients won't propagate here

        log_prob_z = torch.sum(dist.log_prob(z), dim=-1)  # batchwise likelyhood

        log_jac_dets = torch.stack(log_jac_dets, dim=0)
        log_jac_det_sum = torch.sum(log_jac_dets, dim=0)  # batchwise jacobians

        log_prob_z = log_prob_z / d  # scaling for loss in reasonable size
        log_jac_det_sum = log_jac_det_sum / d  # scaling for loss in reasonable size

        return z, log_prob_z, log_jac_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): DxT latent space variable

        Returns:
            torch.Tensor:
        """
        z_complex: torch.Tensor = self.dft(z)

        for layer, f in zip(reversed(self.layers), reversed(self.flips)):
            z_complex = layer.inverse(z_complex, flip=f)

        x = self.dft.inverse(z_complex)

        return x

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample new series from the learn distribution

        Args:
            batch_size (int): number of series to sample

        Returns:
            Tensor: signals in the signal space
        """

        self.eval()

        match self.dist_config["dist"]:
            case "studentt":
                dist = torch.distributions.StudentT(self.df, self.loc, self.scale)
            case "normal":
                dist = torch.distributions.Normal(self.loc, self.scale)
            case "cauchy":
                dist = torch.distributions.Cauchy(self.loc, self.scale)
            case "laplace":
                dist = torch.distributions.Laplace(self.loc, self.scale)

        with torch.no_grad():
            z = dist.rsample((batch_size, self.seq_len)).reshape(
                (batch_size, self.seq_len)
            )

            signals = self.inverse(z)

        return signals
