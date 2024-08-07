from typing import Any, Dict

import torch
import torch.nn as nn
from type_converter import TypeConverter


class FinGan(nn.Module):
    def __init__(
        self,
        gen_config: Dict[str, Any],
        disc_config: Dict[str, Any],
        data_config: Dict[str, Any],
        dist_config: Dict[str, Any],
    ):
        """Time Gan containing 5 networks for training and evaluation

        The training data is supposed to be scaled and shifted,
        i.e.
        training_data = (log_return_data - shift) / scale
        this is important for sampling

        Args:
            gen_config (Dict[str, Any]) : generator settings
                seq_len
                dtype
            disc_conig (Dict[str, Any]) : discriminator settings
                input_dim
                dtype
            data_config (Dict[str, Any]): scale and shift of input / output data (input_data = (log_returns - shift) / scale)
                scale
                shift
            dist_config [Dict[str, Any]]: distribution configuration
                dist : 'normal', 'studentt', 'cauchy'
                params (Dict[str, Any]) : 'loc', 'scale', ('df' if 'studentt')

            scale (float): scale of the data
            shift (shift): shift of the data
        """

        nn.Module.__init__(self)

        self._gen_config = gen_config
        self.gen_config = gen_config.copy()
        self._disc_config = disc_config
        self.disc_config = disc_config.copy()
        self.data_config = data_config
        self.dist_config = dist_config

        if isinstance(self.gen_config["dtype"], str):
            self.gen_config["dtype"] = TypeConverter.str_to_torch(
                self.gen_config["dtype"]
            )

        if isinstance(self.disc_config["dtype"], str):
            self.disc_config["dtype"] = TypeConverter.str_to_torch(
                self.disc_config["dtype"]
            )

        model = self.gen_config["model"]
        del self.gen_config["model"]
        if model == "mlp":
            self.gen = FinGanMLP(**self.gen_config)
        self.disc = FinGanDisc(**self.disc_config)

        # init wiehgts
        self.gen.apply(self._weights_init)
        self.disc.apply(self._weights_init)

        # note that 100 is a fixed value defined in the models
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

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "gen_config": self._gen_config,
            "disc_config": self._disc_config,
            "data_config": self.data_config,
            "dist_config": self.dist_config,
        }

        return dict_

    def _weights_init(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            # apply a uniform distribution to the weights and a bias=0
            nn.init.normal_(module.weight.data, 0.0, 0.06)
            nn.init.constant_(module.bias.data, 0.0)

    def sample(self, batch_size, unnormalize: bool = False) -> torch.Tensor:
        """Generate time series from learned distribution

        Args:
            batch_size (int, optional): Batch size to sample

        Returns:
            torch.Tensor: time series of the size (batch_size x series_length)
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

        noise = dist.rsample((batch_size, 100)).reshape((batch_size, 100))
        gen_sample = self.gen(noise)

        if unnormalize:
            gen_sample = (
                gen_sample * self.data_config["scale"] + self.data_config["shift"]
            )

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
            nn.Linear(input_dim, 2048, dtype=dtype),
            nn.Tanh(),
            nn.Linear(2048, 2048, dtype=dtype),
            nn.Tanh(),
            nn.Linear(2048, seq_len, dtype=dtype),
            nn.Tanh(),
        )
