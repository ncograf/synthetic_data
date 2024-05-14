import torch
import torch.nn as nn


class TGRNNNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layer: int,
        in_dim: int,
        out_dim: int,
        bidirect: bool,
        dtype: torch.dtype,
    ):
        """LSTM embedding transformer

        Args:
            T (int): Lenght of the time series
            hidden_dim (int): hidden dimensions in rnn
            num_layer (int): number of stacked lstm layer
            in_dim (int): input dimensions
            out_dim (int): output dimensions
            bidirect (bool): whether the network should be bidirectional or not
            dtype (torch.dtype): Type to be used for the model
        """

        nn.Module.__init__(self)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirect
        self.dtype = dtype

        self.rnn = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layer,
            batch_first=True,
            bidirectional=self.bidirectional,
            dtype=self.dtype,
        )

        self.lin = nn.Linear(
            hidden_dim * (2 if self.bidirectional else 1),
            self.out_dim,
            dtype=self.dtype,
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through rnn

        Args:
            x (torch.Tensor): DxTxI input tensor (D = batch, T time, I input size)
            h (torch.Tensor, optional): DxTxO teacher forcing variable. Defaults to None

        Returns:
            torch.Tensor: DxTxO output embeddings
        """

        if x.dim() == 1:
            x = x.reshape((1, -1, 1))
        elif x.dim() == 2:
            x = x.reshape((*x.shape, 1))

        # TODO for padded input sizes consider packing x for faster evaluation

        x, _ = self.rnn(x)

        x = self.lin(x)

        x = self.act(x)

        return x
