from abc import abstractmethod
import numpy.typing as npt
from typing import Tuple


class BaseGenerator:
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def fit_model(self, price_data: npt.ArrayLike, **kwargs):
        """Fit local Model model with the given stock market price data

        Args:
            data (npt.ArrayLike): stock market prices for one stock
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )

    @property
    def name(self):
        return self._name

    @abstractmethod
    def check_model(self):
        """Raises an error if no model is available"""
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )

    @abstractmethod
    def generate_data(
        self, len: int = 500, burn: int = 100, **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates synthetic data with the previously trained model

        Args:
            len (int, optional): Number of synthetic data points. Defaults to 500.
            burn (int, optional): Number of data points to neglect before sampling the returned ones. Defaults to 100.
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )
