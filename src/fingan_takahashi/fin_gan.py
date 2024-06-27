from typing import Any, Dict

import torch
import torch.nn as nn
from type_converter import TypeConverter


class FinGan(nn.Module):
    def __init__(
        self,
        gen_config: Dict[str, Any],
        disc_config: Dict[str, Any],
        dtype: str,
        scale: float,
        shift: float,
    ):
        """Time Gan containing 5 networks for training and evaluation

        The training data is supposed to be scaled and shifted,
        i.e.
        training_data = (log_return_data - shift) / scale
        this is important for sampling

        Args:
            gen_config (Dict[str, Any]),
            disc_conig (Dict[str, Any]),
            dtype (str): sets the models dtype
            scale (float): scale of the data
            shift (shift): shift of the data
        """

        nn.Module.__init__(self)

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        self.gen_config = gen_config
        self.disc_config = disc_config
        self.gen_config["dtype"] = self.dtype
        self.disc_config["dtype"] = self.dtype

        self.scale = scale
        self.shift = shift

        self.gen = FinGanMLP(**self.gen_config)
        self.disc = FinGanDisc(**self.disc_config)

        # init wiehgts
        self.gen.apply(self._weights_init)
        self.disc.apply(self._weights_init)

        # note that 100 is a fixed value defined in the models
        self.mean = nn.Parameter(torch.zeros(100), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(100), requires_grad=False)

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "gen_config": self.gen_config,
            "disc_config": self.disc_config,
            "dtype": self.dtype_str,
            "scale": self.scale,
            "shift": self.shift,
        }

        return dict_

    def _weights_init(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            # apply a uniform distribution to the weights and a bias=0
            nn.init.normal_(module.weight.data, 0.0, 0.06)
            nn.init.constant_(module.bias.data, 0.0)

    def sample(self, batch_size: int = 1, unnormalize: bool = False) -> torch.Tensor:
        """Generate time series from learned distribution

        Args:
            batch_size (int, optional): Batch size to sample

        Returns:
            torch.Tensor: time series of the size (batch_size x series_length)
        """

        dist = torch.distributions.MultivariateNormal(self.mean, self.cov)
        noise = dist.rsample((batch_size,))
        gen_sample = self.gen(noise)

        if unnormalize:
            gen_sample = gen_sample * self.scale + self.shift

        return gen_sample

    def forward(self, x):
        self.disc(x)


class FinGanDisc(nn.Sequential):
    def __init__(self, input_dim: int, dtype: torch.dtype):
        in_channels = 1
        input_dim = input_dim
        kernel_size = 10
        nn.Sequential.__init__(
            self,
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(input_dim * 128, 32, dtype=dtype),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, 1, dtype=dtype),
            nn.Sigmoid(),
        )


class FinGanMLP(nn.Sequential):
    def __init__(self, seq_len: int, dtype: torch.dtype):
        input_dim = 100
        nn.Sequential.__init__(
            self,
            nn.Linear(input_dim, 128, dtype=dtype),
            nn.Tanh(),
            nn.Linear(128, 2048, dtype=dtype),
            nn.Tanh(),
            nn.Linear(2048, seq_len, dtype=dtype),
            nn.Tanh(),
        )
