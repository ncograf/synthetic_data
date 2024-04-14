from typing import Tuple

import base_generator
import numpy as np
import numpy.typing as npt
from arch.univariate import GARCH, ConstantMean, Distribution, Normal
from arch.univariate.base import ARCHModelResult


class GarchGenerator(base_generator.BaseGenerator):
    # TODO Make this more efficient

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        distribution: Distribution = Normal(),
        name="GARCH",
    ):
        base_generator.BaseGenerator.__init__(self, name)

        self._model: ConstantMean | None = None
        self._fitted: ARCHModelResult | None = None
        self._distribution = distribution
        self._p = p
        self._q = q

    def fit_model(self, price_data: npt.ArrayLike):
        """Fit Garch model with the given stock market price data

        Args:
            data (npt.ArrayLike): stock market prices for one stock
        """

        if price_data.ndim != 1:
            raise ValueError("Input price data must be one dimensional")

        data = np.array(price_data)
        data_mask = ~np.isnan(data)
        self._zero_price = np.array(data)[data_mask][0]
        returns = (data[1:] / data[:-1]) - 1
        percent_returns = returns * 100
        return_mask = ~np.isnan(percent_returns)

        self._model = ConstantMean(percent_returns[return_mask])
        self._model.volatility = GARCH(p=self._p, q=self._q)
        self._model.distribution = self._distribution
        self._fitted = self._model.fit()

    def model(self) -> ConstantMean:
        return self._model

    def check_model(self):
        if self._model is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self, len: int = 500, burn: int = 100, seed: int = 99
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        np.random.seed(seed)
        return_simulation = (
            np.array(
                self._model.simulate(self._fitted.params, nobs=len, burn=burn).loc[
                    :, "data"
                ],
                dtype=np.float64,
            )
            / 100
            + 1
        )
        price_simulation = np.zeros_like(return_simulation, dtype=np.float64)
        price_simulation[0] = self._zero_price

        for i in range(1, price_simulation.shape[0]):
            price_simulation[i] = price_simulation[i - 1] * return_simulation[i]

        return (price_simulation, return_simulation)
