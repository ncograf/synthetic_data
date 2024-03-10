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
        self._sample_name = "No Name Given"
        self._figure_name = "No Name Given"

    @property
    def name(self):
        return self._name

    def _check_data_validity(self, data : Union[pd.DataFrame, pd.Series]):
        """Checks data to be time in the index and columns to be string if dataframe

        Args:
            data (Union[pd.DataFrame, pd.Series]): data to be checked

        Raises:
            ValueError: If any of the checks does not pass
        """

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
        """Check whether statistics

        Raises:
            ValueError: If statistics is None
        """
        
        if self.statistic is None:
            raise ValueError("Statistics must be computed before calling this function.")
    
    @abstractmethod
    def set_statistics(self, data : Union[pd.DataFrame, pd.Series], **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    @abstractmethod
    def draw_point(self,
                   axes : plt.Axes,
                   point: Tuple[pd.Timestamp, str],
                   **kwargs):
        raise NotImplementedError("This function must be overwritten!")
    
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
        
    def draw_histogram(self,
                          axes : plt.Axes,
                          color : str = "green",
                          density : bool = True,
                          **kwargs):
        """Plot histogram of distribution

        Args:
            axes (plt.Axes): axes to plot onto
            color (str): color in which to plot the histogram
            density (bool): indicator whether desity is t be plotted instead of values

        Raises:
            NotImplementedError: The function currently only works for Series
        """
        
        self._check_statistics()
        
        if not isinstance(self.statistic, pd.Series):
            raise NotImplementedError("Drawing currently only works with series")

        # plot as histogram
        n_bins = np.minimum(self.statistic.shape[0], 150)
        axes.hist(x=self.statistic, bins=n_bins, color=color, density=density)
        axes.set_label("Distribution")
        axes.set_xlabel(self._name)
        axes.set_ylabel("Density" if density else self._sample_name)