from abc import abstractproperty, abstractmethod
from typing import List, Iterable
from tick import Tick
import pandas as pd


class BaseOutlierSet:
    """Outlier Detector class"""

    def __init__(self):
        self._outlier: List[Tick] = None
        self._name = "Not Set Name"

    @abstractmethod
    def get_outlier(self, symbol: str | None = None) -> Iterable[Tick]:
        """Get set of outlier for given symbol or all if symbol is None

        Args:
            symbol (str | None, optional): Symbol to filter. Defaults to None.

        Returns:
            Set[Tick]: Set of ticks considered outlier
        """
        raise NotImplementedError(
            "The set outliers function needs to be implementd before use"
        )

    @abstractmethod
    def set_outlier(self, data: pd.DataFrame):
        """Comptues outliers form given DataFrame

        Args:
            data (pd.DataFrame): data to look for outlier

        """
        raise NotImplementedError(
            "The set outliers function needs to be implementd before use"
        )

    @abstractproperty
    def data(self) -> pd.DataFrame:
        """Data property containing the data on which the outliers are computed"""
        raise NotImplementedError("The data property must be implememnted first")

    def check_outlier(self):
        """Check whether outliers are set

        Raises:
            ValueError: If outlier is None
        """
        if self._outlier is None:
            raise ValueError("Outliler must be computed before calling this function.")

    def is_outlier(self, tick: Tick, **kwargs) -> bool:
        """Check whether point is in the outlier array

        Args:
            point (Tuple[pd.Timestamp  |  str]): point to check

        Raises:
            IndexError: Outlier not yet computed

        Returns:
            bool: Point is in outliers of this statistic
        """
        self.check_outlier()
        return tick in self._outlier
