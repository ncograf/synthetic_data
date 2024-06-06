from typing import Any, Dict

import networks
import torch
import torch.nn as nn
from type_converter import TypeConverter


class FinGan(nn.Module):
    def __init__(
        self,
        gen_config: Dict[str, Any],
        disc_config: Dict[str, Any],
        dtype: str,
    ):
        """Time Gan containing 5 networks for training and evaluation

        Args:
            gen_config (Dict[str, Any]),
            disc_conig (Dict[str, Any]),
            dtype (str): sets the models dtype
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

        # Define necessary networks
        self.gen = networks.model_factory(self.gen_config)
        self.disc = networks.model_factory(self.disc_config)

        self.mean = nn.Parameter(
            torch.zeros(self.gen_config["input_dim"]), requires_grad=False
        )
        self.cov = nn.Parameter(
            torch.eye(self.gen_config["input_dim"]), requires_grad=False
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "gen_config": self.gen_config,
            "disc_config": self.disc_config,
            "dtype": self.dtype_str,
        }

        return dict_

    def sample(self, batch_size: int = 1, burn: int = 0) -> torch.Tensor:
        """Generate time series from learned distribution

        Args:
            batch_size (int): Batch size to sample
            burn (int): ignored

        Returns:
            torch.Tensor: time series of the size (batch_size, series_length, n_stocks)
        """

        dist = torch.distributions.MultivariateNormal(self.mean, self.cov)
        noise = dist.rsample((batch_size,))
        return self.gen(noise)

    def forward(self, x):
        self.disc(x)
