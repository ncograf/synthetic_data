from abc import abstractmethod
from typing import Union, Tuple, Optional, Set, Dict, List
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

class BaseStatistic:

    def __init__(self):
        self.statistic = None # initialize as not but make
        self.point = None # store the last point in the drawn statistics
        self._name = "No Name Given"
        
    @abstractmethod
    def _get_mask(self, **kwargs) -> npt.NDArray:
        raise NotImplementedError("This function must be overwritten!")

    @property
    def name(self):
        return self._name

    def _check_data_validity(self, data : Union[pd.DataFrame, pd.Series]):

        # check data on validity
        if (not isinstance(data, pd.DataFrame)) and (not isinstance(data, pd.Series)):
            raise ValueError("Data must be either series or dataframe.")

        if isinstance(data, pd.DataFrame) and not pd.api.types.is_string_dtype(data.columns.dtype):
            raise ValueError("Data column names must be string.")

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be Timestamps")

        if isinstance(data, pd.Series) and not isinstance(data.name, str):
            raise ValueError("Series name must be string")
    
    def _check_statistics(self):
        
        if self.statistic is None:
            raise ValueError("Statistics must be computed before calling this function.")
    
    @abstractmethod
    def set_statistics(self, data : Union[pd.DataFrame, pd.Series], **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    @abstractmethod
    def draw_distribution(self,
                          axes : plt.Axes,
                          **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    @abstractmethod
    def draw_point(self,
                   axes : plt.Axes,
                   index: Tuple[pd.Timestamp, str],
                   point_array: List[Tuple[pd.Timestamp, str]],
                   **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    def _get_index(self, mask : npt.ArrayLike) -> Set[Tuple[any, any]]:
        """ Compute a list of indices and columns from mask

        Args:
            mask (npt.ArrayLike): mask of which to get the indices

        Returns:
            List[Tuple[any, any]]: List with indices where the mask is true
        """
        
        self._check_statistics()

        if isinstance(self.statistic, pd.DataFrame):
            np_indices = np.where(mask)
            indices = self.statistic.index[np_indices[0]]
            columns = self.statistic.columns[np_indices[1]]
        elif isinstance(self.statistic, pd.Series):
            np_indices = np.where(mask) # transfrom mask to list
            indices = self.statistic.index[np_indices]
            columns = [self.statistic.name] * len(indices)
        else:
            raise ValueError("Unsuported type in statistics. Required to be DataFrame or Series")

        return set(zip(indices, columns))
    
    def get_statistic(self, point : Tuple[pd.Timestamp, str]) -> float:
        """Get data value at the given point

        The data value of this class represents a statistic

        Args:
            point (Tuple[pd.Timestamp, str]): Point in Series (name/str will be ignored) or Dataframe

        Returns:
            float: Statistics
        """

        self._check_statistics()
        
        if isinstance(self.statistic, pd.DataFrame):
            return self.statistic.at[point[0], point[1]]
        elif isinstance(self.statistic, pd.Series):
            return self.statistic.at[point[0]]
        else:
            ValueError("Statistic must be series or DataFrame.")
        
    def get_outliers(self, **kwargs) -> Set[Tuple[pd.Timestamp, str]]:
        """Computes a list of outlier points in the dataframe / series

        If dataframe a list of tuples is returned, otherwise a list of indices

        Args:
            **kwargs (any): arguments which will be passed down to the mask

        Returns:
            List[Tuple[pd.Timestamp, str]]: List with indices where the mask is true
        """
        mask = self._get_mask(**kwargs)
        return self._get_index(mask=mask)