from typing import Dict, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
import univar_garch_model
from arch.univariate import GARCH, ConstantMean, Normal, StudentsT


class GarchUnivarGenerator(base_generator.BaseGenerator):
    def fit_model(
        self, price_data: pd.DataFrame, config: Dict[str, any]
    ) -> Tuple[Dict[str, any], float]:
        """Fit Garch model with the given stock market prices

        For each price a Garch model is fitted and for the sampling
        a copula is fit on top of that.

        To have suitable data, all nan samples are dropped.

        Returns the confguration for the univariate garch model

        Args:
            price_data (pd.DataFrame): stock market prices for one stock
            config (Dict[str, any]): Configuration of the GARCH(p,q) model,
                Must contain the following parameter:
                    - p : int
                    - q : int
                    - dist : literal['normal', 'studentT']

        Returns:
            Tuple[Dict[str,any], float]: garch_configs, initial price
        """

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        if price_data.shape[1] > 1:
            raise ValueError("Price data must be at most one symbol")

        dist = config["dist"]
        p = config["p"]
        q = config["q"]

        if dist == "normal":
            distribution = Normal()  # for univariate garch
        elif dist.lower() == "studentt":
            distribution = StudentsT()  # for univariate garch
        else:
            ValueError("The chosen distribution is not supported")

        # algorithms only work with dataframes
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        # create data
        data = np.array(price_data)
        returns = (data[1:] / data[:-1]) - 1
        percent_returns = returns * 100
        return_mask = ~np.isnan(percent_returns).flatten()
        scaled_returns = percent_returns[return_mask]
        initial_price = data[1:][return_mask][0]

        # fit garch models
        model = ConstantMean(percent_returns[return_mask])
        model.volatility = GARCH(p=p, q=q)
        model.distribution = distribution
        fitted = model.fit()
        scaled_returns = scaled_returns / fitted.conditional_volatility[return_mask]

        # store fitted garch model in minimal format
        garch_config = {}
        garch_config["alpha"] = [fitted.params[f"alpha[{i}]"] for i in range(1, p + 1)]
        garch_config["beta"] = [fitted.params[f"beta[{i}]"] for i in range(1, q + 1)]
        garch_config["mu"] = fitted.params["mu"]
        garch_config["omega"] = fitted.params["omega"]

        if dist.lower() == "studentt":
            garch_config["nu"] = fitted.params["nu"]

        garch_config["dist"] = dist

        out_dict = {"garch": garch_config, "init_price": initial_price}
        return out_dict

    def sample(
        self, length: int, burn: int, config: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from garch copula

        Args:
            length (int): Number of samples.
            burn (int): Number of samples in the beginning to be neglected.
            config (Dict[str, any]): Garch configuration with a dictionary and float
                "garch" : Dict containing the univariate garch configurations
                "init_price": float initial price

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price simulations and return simulations
        """

        garch_dict = config["garch"]
        initial_price = config["init_price"]
        model = univar_garch_model.UnivarGarchModel(
            garch_dict=garch_dict, initial_price=initial_price
        )

        price_simulation, return_simulation = model.sample(
            length=length, burn=burn, dtype=np.float64
        )

        return price_simulation, return_simulation
