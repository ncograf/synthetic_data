from typing import Dict

import base_statistic
import numpy as np
import quantile_statistic
from matplotlib.axes import Axes
from pandas import DataFrame, Series


class TailStatistic(base_statistic.BaseStatistic):
    def __init__(self, underlying: quantile_statistic.QuantileStatistic):
        base_statistic.BaseStatistic.__init__(self)
        self.underlying = underlying

    def set_statistics(self, data: DataFrame | Series, **kwargs):
        self.underlying.check_statistic()
        self._mask = self.underlying._get_mask()
        self._statistic = np.copy(self.underlying.statistic)
        self._statistic[~self._mask] = np.nan

    def draw_histogram(
        self,
        axes: Axes,
        style: Dict[str, any],
        y_label: str = "Density",
        y_log_scale: bool = True,
        **kwargs,
    ):
        axes.clear()  # mak sure everything is redrawn

        # plot as histogram
        if self.statistic.ndim != 2:
            raise RuntimeError("Only one dimensional statistic support histograms")

        _data = self.statistic[self._mask]
        _data = _data[~np.isnan(_data)]

        n_bins = np.minimum(_data.shape[0], 150)
        axes.hist(x=_data, bins=n_bins, **style, log=y_log_scale)
        axes.set_xlabel(self._name)
        axes.set_ylabel(y_label)


class UpperTailStatistic(TailStatistic):
    def __init__(self, underlying: quantile_statistic.QuantileStatistic):
        TailStatistic.__init__(self, underlying=underlying)

    def set_statistics(self, data: DataFrame | Series, **kwargs):
        self.underlying.check_statistic()
        self._mask = self.underlying._get_upper_mask()
        self._statistic = np.copy(self.underlying.statistic)
        self._statistic[~self._mask] = np.nan


class LowerTailStatistic(TailStatistic):
    def __init__(self, underlying: quantile_statistic.QuantileStatistic):
        TailStatistic.__init__(self, underlying=underlying)

    def set_statistics(self, data: DataFrame | Series, **kwargs):
        self.underlying.check_statistic()
        self._mask = self.underlying._get_lower_mask()
        self._statistic = np.copy(self.underlying.statistic)
        self._statistic[~self._mask] = np.nan
