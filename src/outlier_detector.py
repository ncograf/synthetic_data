from abc import abstractmethod, abstractproperty
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Set
import pandas as pd
import numpy as np
import numpy.typing as npt


class OutlierDetector:
    """Outlier Detector class"""

    @abstractproperty
    def data(self) -> Union[pd.Series, pd.DataFrame]:
        """Data property containing the data on which the outliers are computed"""
        raise NotImplementedError("The data property must be implememnted first")
    
    @abstractmethod
    def get_outliers(self, **kwargs) -> Set[Tuple[pd.Timestamp, str]]:
        """Computes a list of outlier points

        Args:
            **kwargs (any): any arguments used for the comuptation

        Returns:
            List[Tuple[pd.Timestamp, str]]: List of outliers points in the time series
        """
        raise NotImplementedError("The get outliers function needs to be implementd before use")
    
    @abstractmethod
    def is_outlier(self, point : Tuple[pd.Timestamp, str]) -> bool:
        """Check if a given point is considered outlier according to the detector

        Args:
            point (Tuple[pd.Timestamp, str]): Poit to be checked

        Returns:
            bool: True if the point is an outlier
        """
        raise NotImplementedError("The get outliers function needs to be implementd before use")