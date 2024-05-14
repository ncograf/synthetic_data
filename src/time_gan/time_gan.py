from typing import Any, Dict

import torch
import torch.nn as nn
from tg_rnn_network import TGRNNNetwork
from type_converter import TypeConverter


class TimeGan(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_layer: int,
        n_stocks: int,
        dtype: str = "float32",
    ):
        """Time Gan containing 5 networks for training and evaluation

        Args:
            hidden_dim (int): hidden dimensions in rnn
            embed_dim (int): hidden dimensions of embeddings
            num_layer (int): number of stacked lstm layer
            n_stocks (int): input dimensions (number of stock to processe simultaneously)
            dtype (str, optional): sets the models dtype
        """

        nn.Module.__init__(self)

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.dtype = TypeConverter.str_to_torch(self.dtype_str)

        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_stocks = n_stocks

        # Define necessary networks
        self.embedder = TGRNNNetwork(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            in_dim=n_stocks,
            out_dim=embed_dim,
            bidirect=False,
            dtype=self.dtype,
        )
        self.recovery = TGRNNNetwork(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            in_dim=embed_dim,
            out_dim=n_stocks,
            bidirect=False,
            dtype=self.dtype,
        )
        # The supervisor is a hack to make the teacher forcing work
        self.supervisor = TGRNNNetwork(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            in_dim=embed_dim,
            out_dim=embed_dim,
            bidirect=False,
            dtype=self.dtype,
        )
        self.generator = TGRNNNetwork(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            in_dim=embed_dim,
            out_dim=embed_dim,
            bidirect=False,
            dtype=self.dtype,
        )
        self.discriminator = TGRNNNetwork(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            in_dim=embed_dim,
            out_dim=1,
            bidirect=True,
            dtype=self.dtype,
        )

        # hack to get device for sampling
        self.device_param = nn.Parameter(torch.empty(0, dtype=self.dtype))

    @property
    def device(self) -> torch.device:
        return self.device_param.device

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "num_layer": self.num_layer,
            "n_stocks": self.n_stocks,
            "dtype": self.dtype_str,
        }

        return dict_

    def sample_noise(self, B: int, T: int) -> torch.Tensor:
        """Compute noise for the generator input

        Args:
            B (int): Batch size
            T (int): Time series length

        Returns:
            torch.Tensor: Sampled noise tensor of size (B, T, embed_dim)
        """
        noise = torch.rand(
            size=(B, T, self.embed_dim), device=self.device, dtype=self.dtype
        )
        return noise

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Call embedder network to get from returns to embeddings

        Args:
            x (torch.Tensor): time series of dimension (#Batch, #sequence, n_stocks)

        Returns :
            torch.Tensor: embeddings of dimension (#Batch, #sequence, embed_dim)

        """
        return self.embedder(x)

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        """Call recovery network to get from embedding sequences to returns

        Args:
            x (torch.Tensor): embeddings of dimension (#Batch, #sequence, embed_dim)

        Returns :
            torch.Tensor: time series of dimension (#Batch, #sequence, n_stocks)

        """
        return self.recovery(x)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next element for a given seqences in the embedding space

        This basically applies the supervisor

        Args:
            x (torch.Tensor): input sequence of dimension (#Batch, #sequence, embed_dim)

        Returns :
            torch.Tensor: prediction sequence of dimension (#Batch, #sequence, embed_dim)
        """
        return self.supervisor(x)

    def generate_embed(self, batch_size: int, series_length: int) -> torch.Tensor:
        """Generate sampled embeddings from the learned distribution

        Args:
            batch_size (int): Batch size to sample
            series_length (int): series lenght to sample

        Returns:
            torch.Tensor: embeddings of the size (batch_size, series_length, embed_dim)
        """

        z = self.sample_noise(batch_size, series_length)
        return self.generator(z)

    def discriminate_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate embedings with the discriminator network

        Args:
            x (torch.Tensor): Embedding tensor to be discriminated

        Returns:
            torch.Tensor: tensor whether the the series is real dim (x.shape[0], x.shape[1], 1)
        """

        return self.discriminator(x)

    def sample(
        self, series_length: int, batch_size: int = 1, burn: int = 0
    ) -> torch.Tensor:
        """Generate time series from learned distribution

        Args:
            series_length (int): series lenght to sample
            batch_size (int): Batch size to sample
            burn (int): number of predictions to throw away in the beginning

        Returns:
            torch.Tensor: time series of the size (batch_size, series_length, n_stocks)
        """

        embeddings = self.generate_embed(
            batch_size=batch_size, series_length=series_length + burn
        )
        embeddings = self.predict_next(embeddings)
        time_series = self.recover(embeddings)[:, burn:, :]

        return time_series
