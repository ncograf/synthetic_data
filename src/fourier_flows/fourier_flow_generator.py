from typing import List, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import torch
from fourier_flow import FourierFlow


class FourierFlowGenerator(base_generator.BaseGenerator):
    def __init__(
        self,
        name="FourierFlow",
    ):
        base_generator.BaseGenerator.__init__(self, name)

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

        data = np.array(price_data)
        data_mask = ~np.isnan(data)
        data = data[
            data_mask
        ]  # drop the nans TODO, this might give wrong results for missing nans
        self._zero_price = data[0]
        log_returns = np.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(np.swapaxes(X[:, ::lag], 0, 1))

        n_epochs = 1000
        learning_rate = 1e-3

        self._model, self._losses = self.train_fourier_flow(
            X=X, epochs=n_epochs, learning_rate=learning_rate
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

        return_simulation = self._model.sample(len)

        price_simulation = np.zeros_like(return_simulation, dtype=np.float64)
        price_simulation[0] = self._zero_price

        for i in range(1, price_simulation.shape[0]):
            price_simulation[i] = price_simulation[i - 1] * return_simulation[i]

        return (price_simulation, return_simulation)

    def train_fourier_flow(
        self, X: torch.Tensor, epochs: int, learning_rate: float
    ) -> Tuple[FourierFlow, List[float]]:
        D = X.shape[0]
        T = X.shape[1]
        hidden_dim = T * 2
        n_layer = 10

        model = FourierFlow(hidden_dim=hidden_dim, D=D, T=T, n_layer=n_layer)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            z, log_prob_z, log_jac_det = model(X)
            loss = torch.mean(-log_prob_z - log_jac_det)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0:
                print(
                    (
                        f"Epoch: {epoch:>10d}, last loss {loss.item():>10.4f},"
                        f"aveage_loss {np.mean(losses):>10.4f}"
                    )
                )

        print("Finished training!")

        return model, losses
