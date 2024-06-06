from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from conditional_flow import ConditionalFlow
from torch.distributions.multivariate_normal import MultivariateNormal
from type_converter import TypeConverter


class RegimeCondFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim: int,
        conditional_dim: int,
        n_layer: int,
        num_model_layer: int,
        drop_out: float,
        activation: str,
        regime_config: Dict[str, Any],
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
            regime_config (Dict[str, Any]): Configuration for regime detection
            norm (Literal['layer', 'batch', 'none']): normalization layer to be used.
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
            dft_scale (float, optional): Amount to scale dft signal. Defaults to 1.
            dft_shift (float, optional): Amount to shift dft signal. Defaults to 0.
        """

        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        self.dim = dim
        self.conditional_dim = conditional_dim
        self.n_layer = n_layer
        self.num_model_layer = num_model_layer
        self.drop_out = drop_out
        self.activation = activation
        self.norm = norm
        self.dtype = dtype
        self.dft_scale = dft_scale
        self.dft_shift = dft_shift
        self.regime_detector = BaseRegimeDetector(**regime_config)

        self.eye = nn.Parameter(
            torch.eye(dim, dtype=TypeConverter.str_to_torch(self.dtype)),
            requires_grad=False,
        )
        self.zero = nn.Parameter(
            torch.zeros(dim, dtype=TypeConverter.str_to_torch(self.dtype)),
            requires_grad=False,
        )

        n_regimes = 2
        self.regimes = []
        for i in range(n_regimes):
            self.regimes.append(
                ConditionalFlow(
                    hidden_dim=hidden_dim,
                    dim=dim,
                    conditional_dim=conditional_dim,
                    n_layer=n_layer,
                    num_model_layer=num_model_layer,
                    drop_out=drop_out,
                    activation=activation,
                    norm=norm,
                    dtype=dtype,
                    dft_scale=dft_scale,
                    dft_shift=dft_shift,
                )
            )

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
            "regime_config": self.regime_detector.get_detector_info(),
            "norm": self.norm,
            "dtype": self.dtype,
            "dft_scale": self.dft_scale,
            "dft_shift": self.dft_shift,
        }

        return dict_

    def set_normilizing(self, X: torch.Tensor):
        for regime in self.regimes:
            regime.set_normilizing(X)

    def forward(
        self, x: torch.Tensor, regime: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one forward pass of the network

        Args:
            x (torch.Tensor): NxTxD signal batch tensor
            regime (int): Regime of the computation

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: transformed tensor
        """

        return self.regimes[regime](x)

    def inverse(
        self, z: torch.Tensor, x_cond: torch.Tensor, regime: int
    ) -> torch.Tensor:
        """Compute the signal from a latent space variable

        Args:
            z (torch.Tensor): Dxoutput_dim latent space variable
            x_cond (torch.Tensor): Nxhidden_dim conditional tensor
            regime (int): Regime of the inverse computation

        Returns:
            torch.Tensor: return value D dimensional
        """

        return self.regimes[regime].inverse(z, x_cond)

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
            x = x.to(TypeConverter.str_to_torch(self.dtype))
            n_x = x.shape[1]

            dist_z = MultivariateNormal(self.zero, self.eye)
            z = dist_z.rsample(sample_shape=(n,))

            signals = torch.zeros(
                (1, n_x + n, self.dim), dtype=TypeConverter.str_to_torch(self.dtype)
            )
            signals[0, :n_x, :] = x
            for i in range(z.shape[0]):
                start = max(0, n_x + i - 1536)
                current_signal = signals[0:, start : n_x + i, :]
                regime = self.regime_detector.classify(current_signal[0, :])
                signals[0, n_x + i, :] = self.regimes[regime].sample_single(
                    z[i], current_signal
                )

        return signals[0, n_x:, :]


class BaseRegimeDetector:
    def __init__(self, thresh: float | None):
        self.thresh = thresh

    def fit(self, data: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """Fits a Regime Detector

        Args:
            data (torch.Tensor): data samples x seq_len x stocks  to used for training

        Returns:
            List[torch.Tensor]: Training set indices
        """

        abs_values = torch.nanmean(torch.abs(data), axis=(1, 2))

        indices = torch.arange(abs_values.shape[0])
        self.thresh = torch.median(abs_values)
        regime0 = indices[abs_values <= self.thresh]
        regime1 = indices[abs_values > self.thresh]

        return [regime0, regime1]

    def get_detector_info(self) -> Dict[str, Any]:
        """Initialization parameters for an equivalent detector

        Returns:
            Dict[str, Any]: Configuration
        """

        return {"thresh": self.thresh}

    def classify(self, data: torch.Tensor) -> int:
        """Classify datapoint

        Args:
            data (torch.Tensor): seq_len x stocks

        Returns:
            int: regime of the data
        """

        if self.thresh is None:
            raise RuntimeError("The Regime detector needs to be fit before used!")

        abs_value = torch.abs(torch.mean(data[:-1, :], axis=1))
        return int(abs_value > self.thresh)
