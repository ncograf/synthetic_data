from typing import Any, Dict

import torch
import torch.nn as nn


def model_factory(config: Dict[str, Any]) -> nn.Module:
    """Model Factory creating custom models

    Attention:
        The input dimension  and output dimension of LSTM and MLP have different meanings:
            MLP: batch x input_dim -> batch x output_dim
            LSTM: batch x sequence_len [x 1] -> batch x seqence_len (if input_dim = output_dim = 1)
                  or batch x sequence_len x input_dim -> batch x seqence_len (if input_dim > 1 and output_dim = 1)
                  or batch x sequence_len x input_dim -> batch x seqence_len x output_dim (if input_dim > 1 and output_dim > 1)

    Args:
        config (Dict[str, Any]): model configuration
            arch : model architecture
            num_model_layer : number of model layers
            hidden_dim : size of hidden layer
            dtype : torch dtype
            output_dim : output dimension (note different behaviours of MLP and RNN)
            input_dim : input dimension (note different behaviours of RNN)
            drop_out : float in [0,1) rate
            activation : (str) activation functions 'sigmoid', 'tanh', 'relu', 'softplus'
            norm : (optional) choice 'layer', 'batch', 'None' Default: 'None' only used in MLP
            reduction : (int | Literal['mean', 'sum', 'max', 'min', 'none']) reduction used for for rnn.
            bidirect : (optional) only used for rnn models. Defaults to True

    Returns:
        nn.Module: neural net ready to train
    """

    arch = config["arch"]
    hidden_dim = config["hidden_dim"]
    num_model_layer = config["num_model_layer"]
    dtype = config["dtype"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    drop_out = config["drop_out"]
    activation = config["activation"]

    if activation == "relu":
        sigma = nn.ReLU()
    elif activation == "tanh":
        sigma = nn.Tanh()
    elif activation == "softplus":
        sigma = nn.Softplus()
    elif activation == "celu":
        sigma = nn.CELU()
    else:
        sigma = nn.Sigmoid()

    if arch == "MLP":
        norm = None
        if "norm" in config.keys():
            if config["norm"].lower() == "batch":
                norm = nn.BatchNorm1d
            if config["norm"].lower() == "layer":
                norm = nn.LayerNorm

        model_layer = [nn.Linear(input_dim, hidden_dim, dtype=dtype)]

        for _ in range(num_model_layer):
            if drop_out > 0 and drop_out < 1:
                model_layer.append(nn.Dropout1d(drop_out))
            model_layer.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            if norm is not None:
                model_layer.append(norm(hidden_dim, dtype=dtype))
            model_layer.append(sigma)

        model_layer.append(nn.Linear(hidden_dim, output_dim, dtype=dtype))
        net = nn.Sequential(*model_layer)

    if arch == "LSTM":
        if "bidirect" in config.keys():
            bidirect = config["bidirect"]
        else:
            bidirect = True

        if "reduction" in config.keys():
            reduction = config["reduction"]
        else:
            reduction = "none"

        net = LSTM(
            hidden_dim=hidden_dim,
            dtype=dtype,
            num_rnn_layer=num_model_layer,
            bidirect=bidirect,
            drop_out=drop_out,
            output_dim=output_dim,
            input_dim=input_dim,
            reduction=reduction,
        )

    return net


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype,
        drop_out: float,
        reduction: int | str | None,
        num_rnn_layer: int = 3,
        bidirect: bool = True,
        output_dim: int = 1,
        input_dim: int = 1,
    ):
        nn.Module.__init__(self)

        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layer,
            batch_first=True,
            dropout=drop_out,
            bidirectional=bidirect,
            dtype=dtype,
        )

        self.lin = nn.Linear(
            hidden_dim * (2 if bidirect else 1),
            output_dim,
            dtype=dtype,
        )

        if reduction == "none" or isinstance(reduction, int) or reduction is None:
            self.reduction = reduction
        elif reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "min":
            self.reduction = torch.min
        elif reduction == "max":
            self.reduction = torch.max
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

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

        x, _ = self.lstm(x)

        if isinstance(self.reduction, int):
            x = x[:, self.reduction, :]
        elif not self.reduction == "none" and self.reduction is not None:
            x = self.reduction(x, dim=1)

        x = self.lin(x)

        if self.output_dim == 1 and x.ndim == 3:
            # return sequence
            x = x.squeeze(-1)

        return x
