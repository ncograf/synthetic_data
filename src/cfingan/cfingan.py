from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from type_converter import TypeConverter


class CFinGAN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        disc_seq_len: int,
        dist_config: Dict[str, Any],
        dtype: str | torch.dtype = "float32",
    ):
        """Conditional Flow Network

        Args:
            hidden_dim (int): dimension of the hidden layers needs to be even
            preview (int): number of previewed elements must be even
            disc_seq_len (int): seqence length to train the discriminator
            dist_config (dict): distribution configuration for network
            dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
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
        self.disc_seq_len = disc_seq_len

        self.dist_config = dist_config

        self.g = ContextNet(1, hidden_dim, self.context_dim)

        self.mu = MomentNet(self.context_dim, hidden_dim, 1)
        self.sigma = MomentNet(self.context_dim, hidden_dim, 1)
        if "studentt" == self.dist_config["dist"]:
            self.df = MomentNet(self.context_dim, hidden_dim, 1)

        # hack to know device for sampling
        self.param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.gen = nn.ModuleList([self.g, self.mu, self.sigma])
        self.disc = FinGanDisc(input_dim=disc_seq_len, dtype=self.dtype)

        self.apply(self._init_weights)

    def sample_dist(self, x_cond: torch.Tensor) -> torch.Tensor:
        eps = 1e-7 # minimal variance
        match self.dist_config["dist"]:
            case "studentt":
                return (
                    torch.distributions.StudentT(
                        self.df(x_cond), self.mu(x_cond), torch.abs(self.sigma(x_cond)) + eps
                    )
                    .rsample()
                    .flatten(-2)
                )
            case "normal":
                return (
                    torch.distributions.Normal(
                        self.mu(x_cond), torch.abs(self.sigma(x_cond)) + eps
                    )
                    .rsample()
                    .flatten(-2)
                )
            case "cauchy":
                return (
                    torch.distributions.Cauchy(
                        self.mu(x_cond), torch.abs(self.sigma(x_cond)) + eps
                    )
                    .rsample()
                    .flatten(-2)
                )
            case "laplace":
                return (
                    torch.distributions.Laplace(
                        self.mu(x_cond), torch.abs(self.sigma(x_cond)) + eps
                    )
                    .rsample()
                    .flatten(-2)
                )

    def _init_weights(self, module: nn.Module):
        """Initialize weights for module

        This only initalizes linear layers in the network

        Args:
            module (nn.Module): Module to apply it to.
        """

        if isinstance(module, nn.Linear):
            with torch.no_grad():
                nn.init.normal_(module.weight.data, 0, 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "hidden_dim": self.hidden_dim,
            "disc_seq_len": self.disc_seq_len,
            "dist_config": self.dist_config,
            "dtype": self.dtype_str,
        }

        return dict_

    def f_(
        self, x: torch.Tensor, n: int, hc=None
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

        x_len = x.shape[1]

        x_ = x.reshape(*x.shape, 1)
        x_cond, hc = self.g(x_, hc)
        x = self.sample_dist(x_cond)

        # compute the remaining outputs
        x_out = []
        last_x = x[:, -1:]
        for _ in range((n - x_len)):
            x_cond_, hc = self.g.forward(last_x.reshape(*last_x.shape, 1), hc)
            last_x = self.sample_dist(x_cond_)
            x_out.append(last_x)

        # glue everything together
        if len(x_out) > 0:
            x = torch.cat([x] + x_out, dim=-1)

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

        return self.f_(x, x.shape[1], hc)

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
        x = torch.zeros((n, 1)).to(self.dtype).to(self.param.device)

        x = self.f_(x, seq_len + n_burn)

        return x[:, n_burn:]


class ContextNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        nn.Module.__init__(self)

        self.rnn = nn.LSTM(input_dim, hidden_dim, 3, batch_first=True)

        self.linear_one = nn.Linear(hidden_dim, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim)
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


if __name__ == "__main__":
    flow = CFinGAN(10, 10, {"dist": "normal"})
    x = flow.sample(5, 10, 3)
    z = flow.forward(x)

    print(z)
