from typing import List, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import torch
from fourier_flow import FourierFlow
from torch.utils.data import DataLoader, TensorDataset


class FourierFlowGenerator(base_generator.BaseGenerator):
    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        name: str = "FourierFlow",
        dtype: torch.dtype = torch.float64,
    ):
        base_generator.BaseGenerator.__init__(self, name)
        self.data_min = 0
        self.data_amplitude = 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dtype = dtype

    def fit_model(self, price_data: npt.ArrayLike, lag: int = 1, seq_len: int = 101):
        """Fit a Fourier Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (npt.ArrayLike): stock marked price datsa
            lag (int, optional): lag between the indivisual seqences when splitting the price. Defaults to 1.
            seq_len (int, optional): seqence lenght. Defaults to 101.

        Raises:
            ValueError: If the input is no a single price seqence
        """

        if price_data.ndim != 1:
            raise ValueError("Input price data must be one dimensional")

        data = torch.tensor(price_data, dtype=self.dtype)
        data_mask = ~torch.isnan(data)
        data = data[
            data_mask
        ]  # drop the nans TODO, this might give wrong results for missing nans
        self._zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X)

        self._model, self._losses = self.train_fourier_flow(
            X=X, epochs=self.epochs, learning_rate=self.learning_rate
        )

    def model(self) -> FourierFlow:
        return self._model

    def check_model(self):
        if self._model is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self, len: int = 500, burn: int = 100, **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        self.check_model()

        n = len // self._model.T + 1

        model_output = self._model.sample(n)
        log_returns = (model_output * self.data_amplitude) + self.data_min
        log_returns = log_returns.detach().numpy().flatten()
        return_simulation = np.exp(log_returns[:len])

        price_simulation = np.zeros_like(return_simulation, dtype=np.float32)
        price_simulation[0] = self._zero_price

        for i in range(0, price_simulation.shape[0] - 1):
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

        return (price_simulation, return_simulation)

    def min_max_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            torch.Tensor: scaled data
        """

        self.data_min = torch.min(data)
        data_ = data - self.data_min
        self.data_amplitude = torch.max(data_)

        data_ = data_ / self.data_amplitude

        return data_

    def train_fourier_flow(
        self, X: torch.Tensor, epochs: int, learning_rate: float
    ) -> Tuple[FourierFlow, List[float]]:
        D = X.shape[0]
        T = X.shape[1]
        hidden_dim = T * 2
        n_layer = 10

        model = FourierFlow(
            hidden_dim=hidden_dim, D=D, T=T, n_layer=n_layer, dtype=self.dtype
        )

        X_scaled = self.min_max_scaling(X)

        model.set_normilizing(X_scaled)

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(dataset=dataset, batch_size=128)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()

                z, log_prob_z, log_jac_det = model(batch)
                loss = torch.mean(-log_prob_z - log_jac_det)

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            epoch_loss / len(loader)

            losses.append(epoch_loss)

            if epoch % 10 == 0:
                print(
                    (
                        f"Epoch: {epoch:>10d}, last loss {epoch_loss:>10.4f},"
                        f"aveage_loss {np.mean(losses):>10.4f}"
                    )
                )

        print("Finished training!")

        return model, losses
