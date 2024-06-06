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

    # store keys for convenience
    conf_keys = set([k.lower() for k in config.keys()])

    # architecture must be present in the model description
    arch = config["arch"]

    # get standard elements
    dtype = config["dtype"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    drop_out = config["drop_out"] if "drop_out" in conf_keys else 0

    match config["activation"]:
        case "relu":
            sigma = nn.ReLU()
        case "tanh":
            sigma = nn.Tanh()
        case "softplus":
            sigma = nn.Softplus()
        case "celu":
            sigma = nn.CELU()
        case {"leaky_relu": slope}:
            sigma = nn.LeakyReLU(slope)
        case _:
            sigma = nn.Sigmoid()

    if arch.lower() == "mlp":
        if set(["hidden_dim", "num_model_layer"]) <= conf_keys:
            layers = [config["hidden_dim"]] * config["num_model_layer"]
        elif "layers" in conf_keys:
            layers = config["layers"]

        # check norm
        norm = None
        if "norm" in conf_keys:
            if config["norm"].lower() == "batch":
                norm = nn.BatchNorm1d
            if config["norm"].lower() == "layer":
                norm = nn.LayerNorm

        model_layer = [nn.Linear(input_dim, layers[0], dtype=dtype)]

        drop_out = config["drop_out"] if "drop_out" in conf_keys else 0
        for i in range(1, len(layers)):
            if drop_out > 0 and drop_out < 1:
                model_layer.append(nn.Dropout1d(drop_out))
            model_layer.append(nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            if norm is not None:
                model_layer.append(norm(layers[i], dtype=dtype))
            model_layer.append(sigma)

        model_layer.append(nn.Linear(layers[-1], output_dim, dtype=dtype))
        net = nn.Sequential(*model_layer)

    elif arch.lower() == "lstm":
        # check for optional arguments
        bidirect = config["bidirect"] if "bidirect" in conf_keys else True
        reduction = config["reduction"] if "reduction" in conf_keys else "none"

        net = LSTM(
            hidden_dim=config["hidden_dim"],
            dtype=config["dtype"],
            num_rnn_layer=config["num_model_layer"],
            bidirect=bidirect,
            drop_out=config["drop_out"],
            output_dim=config["output_dim"],
            input_dim=config["input_dim"],
            reduction=reduction,
        )

    elif arch.lower() == "fin_gan_disc":
        in_channels = 1
        input_dim = config["input_dim"]
        kernel_size = 16
        padding = ((kernel_size - 1) // 2,)
        net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                stride=8,
                padding=padding,
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            # seq len input_dim // 8
            sigma,
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                stride=8,
                padding=padding,
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            # seq len input_dim // 64
            sigma,
            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                stride=8,
                padding=padding,
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            # seq len input_dim // 512
            sigma,
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                stride=8,
                padding=padding,
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            # seq len input_dim // 2048
            sigma,
            nn.Flatten(),
            nn.Linear((input_dim // 4096) * 32, 32, dtype=dtype),
            sigma,
            nn.Dropout(drop_out),
            nn.Linear(32, 1, dtype=dtype),
            nn.Sigmoid(),
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
