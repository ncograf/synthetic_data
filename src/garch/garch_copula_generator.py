from typing import Dict, Tuple

import base_generator
import copula_garch_model
import numpy as np
import numpy.typing as npt
import pandas as pd
from arch.univariate import GARCH, ConstantMean, Normal, StudentsT
from arch.univariate.base import ARCHModelResult
from copulas import multivariate, univariate


class GarchCopulaGenerator(base_generator.BaseGenerator):
    def fit_model(
        self, price_data: pd.DataFrame, config: Dict[str, any]
    ) -> Dict[str, any]:
        """Fit Garch copula model with the given stock market prices

        For each price a Garch model is fitted and for the sampling
        a copula is fit on top of that.

        To have suitable data, all rows, containing any nans are dropped.

        Returns the confguration for the copula and the univariate garch models

        Args:
            price_data (pd.DataFrame): stock market prices for one stock
            config (Dict[str, any]): Configuration of the GARCH(p,q) model,
                Must contain the following parameter:
                    - p : int
                    - q : int
                    - dist : literal['normal', 'studentT']

        Returns:
            Dict[str, any]: 'copula' : copula_config, 'garch' : garch_configs, 'init_prices' : initial prices
        """

        dist = config["dist"]
        p = config["p"]
        q = config["q"]

        if dist.lower() == "normal":
            distribution = Normal()  # for univariate garch
            maringal_dist = univariate.GaussianUnivariate()  # for copula
        elif dist.lower() == "studentt":
            distribution = StudentsT()  # for univariate garch
            maringal_dist = univariate.StudentTUnivariate()  # used copula
        else:
            ValueError("The chosen distribution is not supported")

        # algorithms only work with dataframes
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        # create data
        data = np.array(price_data)
        returns = (data[1:] / data[:-1]) - 1
        percent_returns = returns * 100
        return_mask = ~np.isnan(percent_returns)
        nan_row_mask = np.any(~return_mask, axis=1)
        scaled_returns = percent_returns[~nan_row_mask, :]
        initial_price = data[1:, :][~nan_row_mask, :][0]

        # fit garch models
        models = {}
        log_likelyhoods = []
        fitted: Dict[str, ARCHModelResult] = {}
        for i, col in enumerate(price_data.columns):
            models[col] = ConstantMean(percent_returns[return_mask[:, i], i])
            models[col].volatility = GARCH(p=p, q=q)
            models[col].distribution = distribution
            fitted[col] = models[col].fit(disp="off")
            scaled_returns[:, i] = (
                scaled_returns[:, i]
                / fitted[col].conditional_volatility[~nan_row_mask[return_mask[:, i]]]
            )
            log_likelyhoods.append(fitted[col].loglikelihood)

        # store fitted garch model in minimal format
        garch_config = {"alpha": {}, "beta": {}, "mu": {}, "omega": {}, "nu": {}}
        for sym in fitted.keys():
            garch_config["alpha"][sym] = [
                fitted[sym].params[f"alpha[{i}]"] for i in range(1, p + 1)
            ]
            garch_config["beta"][sym] = [
                fitted[sym].params[f"beta[{i}]"] for i in range(1, q + 1)
            ]
            garch_config["mu"][sym] = fitted[sym].params["mu"]
            garch_config["omega"][sym] = fitted[sym].params["omega"]

            if dist.lower() == "studentt":
                garch_config["nu"][sym] = fitted[sym].params["nu"]

        # fit copula with scaled returns
        scaled_returns_df = pd.DataFrame(
            data=scaled_returns,
            columns=price_data.columns,
            index=price_data.iloc[1:].loc[~nan_row_mask].index,
        )
        self.copula = multivariate.GaussianMultivariate(distribution=maringal_dist)
        self.copula.fit(scaled_returns_df)

        # store copula in minimal format
        copula_config = self.copula.to_dict()

        all_config = {
            "copula": copula_config,
            "garch": garch_config,
            "init_prices": initial_price,
            "fit_score": log_likelyhoods,
        }
        return all_config

    def sample(
        self, length: int, burn: int, config: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from garch copula

        Args:
            length (int): Number of samples.
            burn (int): Number of samples in the beginning to be neglected.
            config (Dict[str, any]): Garch copula configuration with two dictionaries and an arraylike
                "copula" : Dict containing copula config
                "garch" : Dict containing the univariate garch configurations
                "init_prices": Arraylike initial prices

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price simulations and return simulations
        """

        copula_dict = config["copula"]
        garch_dict = config["garch"]
        initial_prices = config["init_prices"]
        model = copula_garch_model.CopulaGarchModel(
            copula_dict=copula_dict, garch_dict=garch_dict, initial_price=initial_prices
        )

        price_simulation, return_simulation = model.sample(
            length=length, burn=burn, dtype=np.float64
        )

        return price_simulation, return_simulation
