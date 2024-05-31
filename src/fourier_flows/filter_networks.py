from typing import Any, Dict

import torch
import torch.nn as nn


def model_factory(config: Dict[str, Any]) -> nn.Module:
    """Model Factory creating custom models

    Args:
        config (Dict[str, Any]): model configuration
            arch : model architecture
            num_model_layer : number of model layers
            dtype : torch dtype
            seq_len : sequence lenght
            bidirect : (optional) only used for rnn models
            n_stocks : (optional) only used for rnn models


    Returns:
        nn.Module: neural net ready to train
    """

    arch = config["arch"]
    hidden_dim = config["hidden_dim"]
    num_model_layer = config["num_model_layer"]
    dtype = config["dtype"]
    seq_len = config["seq_len"]
    split_size = seq_len // 2 + 1

    if arch == "MLP":
        model_layer = [nn.Linear(split_size, hidden_dim, dtype=dtype)]

        for _ in range(num_model_layer):
            model_layer.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            model_layer.append(nn.Sigmoid())

        model_layer.append(nn.Linear(hidden_dim, split_size, dtype=dtype))
        net = nn.Sequential(*model_layer)

    if arch == "LSTM":
        if "bidirect" in config.keys():
            bidirect = config["bidirect"]
        else:
            bidirect = True

        if "n_stocks" in config.keys():
            n_stocks = config["n_stocks"]
        else:
            n_stocks = 1

        net = LSTM(
            hidden_dim=hidden_dim,
            dtype=dtype,
            num_rnn_layer=num_model_layer,
            bidirect=bidirect,
            n_stocks=n_stocks,
        )

    return net


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype,
        num_rnn_layer: int = 3,
        bidirect: bool = True,
        n_stocks: int = 1,
    ):
        nn.Module.__init__(self)

        self.n_stocks = n_stocks

        self.lstm = nn.LSTM(
            input_size=n_stocks,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layer,
            batch_first=True,
            bidirectional=bidirect,
            dtype=dtype,
        )

        self.lin = nn.Linear(
            hidden_dim * (2 if bidirect else 1),
            n_stocks,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through rnn

        Args:
            x (torch.Tensor): DxTxI input tensor (D = batch, T time, I input size)

        Returns:
            torch.Tensor: DxTxO output embeddings
        """

        if x.dim() == 1:
            x = x.reshape((1, -1, 1))
        elif x.dim() == 2:
            x = x.reshape((*x.shape, 1))

        # TODO for padded input sizes consider packing x for faster evaluation

        x, _ = self.lstm(x)
        x = self.lin(x)

        if self.n_stocks == 1:
            x = x.squeeze(-1)

        return x
