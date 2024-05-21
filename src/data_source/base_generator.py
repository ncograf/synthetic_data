from abc import abstractmethod
from typing import Tuple

import numpy.typing as npt
import pandas as pd


class BaseGenerator:
    @abstractmethod
    def fit_model(self, price_data: pd.DataFrame, **kwargs):
        """Fit local Model model with the given stock market price data

        Args:
            price_data (npt.ArrayLike): stock market prices for one stock
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )

    @abstractmethod
    def sample(self, len: int, burn: int, **kwargs) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates synthetic data with the previously trained model

        Args:
            len (int): Number of synthetic data points.
            burn (int): Number of data points to neglect before sampling the returned ones.
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )
