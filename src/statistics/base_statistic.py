from abc import abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from tick import Tick
import scipy.stats as ss


class BaseStatistic:
    def __init__(self):
        self._statistic: npt.NDArray | None = None  # initialize as not but make
        if not hasattr(self, "_name"):
            self._name = "No Name Given"
        if not hasattr(self, "_dates"):
            self._dates: List[pd.Timestamp] | None = None
        if not hasattr(self, "_symbols"):
            self._symbols: List[str] | None = None
        if not hasattr(self, "_sample_name"):
            self._sample_name = "No Name Given"
        if not hasattr(self, "_figure_name"):
            self._figure_name = "No Name Given"
        if not hasattr(self, "_figure_name"):
            self._plot_color = "green"

    @property
    def name(self) -> str:
        return self._name

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @property
    def dates(self) -> List[pd.Timestamp]:
        return self._dates

    @property
    def statistic(self) -> npt.NDArray:
        return self._statistic

    def check_data_validity(self, data: pd.DataFrame | pd.Series):
        """Checks data to be time in the index and columns to be string if dataframe

        Args:
            data  (pd.DataFrame | pd.Series): data to be checked

        Raises:
            ValueError: If data does not have the appropriate format
        """

        # check data on validity
        if (not isinstance(data, pd.DataFrame)) and (not isinstance(data, pd.Series)):
            raise ValueError("Data must be either series or dataframe.")

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if not pd.api.types.is_string_dtype(data.columns.dtype):
            raise ValueError("Data column names must be string.")

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be Timestamps")

    def check_statistic(self):
        """Ensures that statistic is set and at least two dimensional

        Raises:
            ValueError: Raises error if statistic is not set
        """
        if self._statistic is None:
            raise ValueError("Statistics must be computed before being referenced.")

        if not self._statistic.ndim >= 2:
            raise ValueError(
                "Statistic must have at least tow axis (columns are symbols and rows timestamps)"
            )

    @abstractmethod
    def set_statistics(self, data: pd.DataFrame | pd.Series, **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    @abstractmethod
    def draw_point(self, axes: plt.Axes, point: Tick, **kwargs):
        raise NotImplementedError("This function must be overwritten!")

    def get_statistic(self, point: Tick) -> npt.ArrayLike:
        """Get data value at the given point

        The data value of this class represents a statistic

        Args:
            point (Tuple[pd.Timestamp, str]): Point in Series (name/str will be ignored) or Dataframe

        Returns (array_like): Point statistics (might be multidimensional)

        """
        row = self.get_date_index(point.date)
        col = self.get_symbol_index(point.symbol)
        return self.statistic[row, col]

    def get_symbol_index(self, column: str | Tick) -> int:
        if isinstance(column, Tick):
            return self._symbols.index(column.symbol)
        if isinstance(column, str):
            return self._symbols.index(column)
        raise RuntimeError("Columns must be either string or Tick")

    def get_date_index(self, date: pd.Timestamp | Tick) -> int:
        if isinstance(date, Tick):
            return self._dates.index(date.date)
        if isinstance(date, pd.Timestamp):
            return self._dates.index(date)
        raise RuntimeError("Date must be either timestamp or Tick")

    def get_kde_all(self) -> ss.gaussian_kde:
        self.check_statistic()
        kde = ss.gaussian_kde(self.statistic[~np.isnan(self.statistic)])
        return kde

    def draw_histogram(
        self,
        axes: plt.Axes,
        symbol: str,
        style: Dict[str, any] = {
            "color": "green",
            "density": True,
        },
        y_label: str = "Density",
        y_log_scale: bool = True,
        **kwargs,
    ):
        """Plot histogram of distribution and cleans old histogram

        Args:
            axes (plt.Axes): axes to plot onto
            symbol (str): stock to plot histogram
            style (Dict[str, any]): styles for plotting
            y_label (bool, optional): Label used for plot. Defaults to 'Density'.
            y_loc_scale (bool, optional): Whether to scale y axis by lograrithmus. Default to True.

        Raises:
            NotImplementedError: The function currently only works for Series
        """

        axes.clear()  # mak sure everything is redrawn

        # plot as histogram
        if self.statistic.ndim != 2:
            raise RuntimeError("Only one dimensional statistic support histograms")

        col = self.get_symbol_index(symbol)

        _data = self.statistic[:, col]
        _data = _data[~np.isnan(_data)]

        n_bins = np.minimum(_data.shape[0], 150)
        axes.hist(x=_data, bins=n_bins, **style, log=y_log_scale)
        axes.set_xlabel(self._name)
        axes.set_ylabel(y_label)

    def draw_flipted_cfd(
        self,
        ax: plt.Axes,
        style: Dict[str, any] = {
            "alpha": 1,
            "marker": "o",
            "markersize": 1,
            "linestyle": "None",
        },
    ):
        """Draws the 1 - cfd(X) where X is over all axes all symbols on the given axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """

        if "color" not in style.keys():
            style["color"] = self._plot_color

        self.check_statistic()
        data = self.normalized_stat()
        ax.plot(data[:, 0], data[:, 1], **style)
