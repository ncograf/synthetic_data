from typing import Dict, Literal, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
from arch.univariate import GARCH, ConstantMean, Normal, StudentsT
from arch.univariate.base import ARCHModelResult
from copulas import multivariate, univariate


class GarchCopulaGenerator(base_generator.BaseGenerator):
    # TODO Make this more efficient

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        distribution: Literal["normal", "studentT"] = "studentT",
        name="GARCHCopula",
    ):
        base_generator.BaseGenerator.__init__(self, name)

        self._models: Dict[str, ConstantMean] | None = None
        self._fitted: Dict[str, ARCHModelResult] | None = None

        if distribution == "normal":
            self._distribution = Normal()
            self._maringal_dist = univariate.GaussianUnivariate()
        elif distribution == "studentT":
            self._distribution = StudentsT()
            self._maringal_dist = univariate.StudentTUnivariate()
        else:
            ValueError("The chosen distribution is not supported")

        self._p = p
        self._q = q

        self.copula = None

    def fit_model(self, price_data: pd.DataFrame):
        """Fit Garch copula model with the given stock market prices

        For each price a Garch model is fitted and for the sampling
        a copula is fit on top of that

        Args:
            data (pd.DataFrame): stock market prices for one stock
        """

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        data = np.array(price_data)
        returns = (data[1:] / data[:-1]) - 1
        percent_returns = returns * 100
        return_mask = ~np.isnan(percent_returns)
        nan_row_mask = np.any(~return_mask, axis=1)
        scaled_returns = percent_returns[~nan_row_mask, :]
        self._zero_price = data[1:, :][~nan_row_mask, :][0]

        self._models = {}
        self._fitted = {}
        for i, col in enumerate(price_data.columns):
            self._models[col] = ConstantMean(percent_returns[return_mask[:, i], i])
            self._models[col].volatility = GARCH(p=self._p, q=self._q)
            self._models[col].distribution = self._distribution
            self._fitted[col] = self._models[col].fit()
            scaled_returns[:, i] = (
                scaled_returns[:, i]
                / self._fitted[col].conditional_volatility[
                    ~nan_row_mask[return_mask[:, i]]
                ]
            )

        scaled_returns_df = pd.DataFrame(
            data=scaled_returns,
            columns=price_data.columns,
            index=price_data.iloc[1:].loc[~nan_row_mask].index,
        )

        self.copula = multivariate.GaussianMultivariate(
            distribution=self._maringal_dist
        )
        self.copula.fit(scaled_returns_df)

    def model(self) -> Dict[str, ConstantMean]:
        return self._models

    def check_model(self):
        if self._models is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self, lenght: int = 500, burn: int = 100, seed: int | None = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from garch copula

        Args:
            lenght (int, optional): Number of samples. Defaults to 500.
            burn (int, optional): Number of samples in the beginning to be neglected. Defaults to 100.
            seed (int | None, optional): Seed for sampling. Defaults to None.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price simulations and return simulations
        """

        if seed is not None:
            np.random.seed(seed)

        copula_sample = np.array(self.copula.sample(lenght + burn))
        cols = self.copula.columns

        return_simulation = np.zeros((lenght, len(cols)), dtype=np.float64)
        price_simulation = np.zeros((lenght + 1, len(cols)), dtype=np.float64)
        price_simulation[0] = self._zero_price

        mu = np.array([self._fitted[col].params["mu"] for col in cols])
        omega = np.array([self._fitted[col].params["omega"] for col in cols])
        alpha = np.array([self._fitted[col].params["alpha[1]"] for col in cols])
        beta = np.array([self._fitted[col].params["beta[1]"] for col in cols])

        var_t = omega / (1 - (alpha + beta))

        for i in range(burn):
            z = copula_sample[i, :]
            epsilon_t = np.sqrt(var_t) * z
            var_t = omega + alpha * epsilon_t**2 + beta * var_t

        for i in range(lenght):
            z = copula_sample[i + burn, :]
            epsilon_t = np.sqrt(var_t) * z
            return_simulation[i, :] = mu + epsilon_t
            var_t = omega + alpha * epsilon_t**2 + beta * var_t
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

        return price_simulation, return_simulation
