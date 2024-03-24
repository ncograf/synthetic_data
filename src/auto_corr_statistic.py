import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional, Literal
from icecream import ic
import matplotlib.pyplot as plt
import temporal_statistc
import scipy.linalg as linalg
import stylized_fact
import boosted_stats

class AutoCorrStatistic(stylized_fact.StylizedFact):
    
    def __init__(
            self,
            max_lag : int,
            underlaying : temporal_statistc.TemporalStatistic,
            legend_postfix : str = '',
            color : str = 'blue', 
            implementation : Literal['boosted', 'strides', 'pyhton_loop'] = 'strides'):
        
        stylized_fact.StylizedFact.__init__(self)

        self._name = r"Auto correlation $\displaymath\frac{\mathbb{E}[(r_{t+k} - \mu)(r_t - \mu)]}{\sigma^2}$"
        self._sample_name = underlaying.name
        self._figure_name = "auto_correlation"
        self._max_lag = max_lag
        self._underlaying = underlaying
        self._plot_color = 'blue'
        self.y_label = 'lag k'

        if implementation == 'strides':
            self.set_statistics = self._set_statistics_stride
        elif implementation == 'boosted':
            self.set_statistics = self._set_statistics_boosted
        elif implementation == 'python_loop':
            self.set_statistics = self._set_statistics_python_loop
        else:
            raise ValueError("The given implementation argument does not exist")
    
    def _set_statistics_stride(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        # compute the means / var for each stock
        mu = np.nanmean(base, axis=0)
        var = np.nanvar(base, axis=0)
        
        # extend the size of the 'log_returns' to be able to crete an appropriate view
        centered = base - mu
        padded = np.pad( centered, 
            ((self._max_lag,0),(0,0)), # only pad the rows 
            mode='constant', constant_values=0)[:-1, :]
        
        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately

        centered_mask = np.isnan(base)
        centered[centered_mask] = 0
        centered_shifted = np.lib.stride_tricks.sliding_window_view(padded, window_shape=centered.shape[0], axis=0, writeable=True)
        centered_shifted = np.swapaxes(centered_shifted, 0,2)
        centered_shifted[np.isnan(centered_shifted)] = 0

        stat = np.einsum('rck,rc->ck', centered_shifted, centered)
        num_entries = np.ones_like(stat) * base.shape[0]
        num_entries[:] -= np.flip(np.arange(1,self._max_lag + 1)) # add the missing values because of the lag offest

        num_missing = np.sum(centered_mask, axis = 0)
        num_entries.T[:] -= num_missing # add missing value because of missing data
        stat = (stat / num_entries)
        stat.T[:] /= var
        self._statistic = np.flip(stat.T, axis=0)
        # TODO compute outlier if needed
        
    def _set_statistics_python_loop(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        # compute the means / var for each stock
        mu = np.nanmean(base, axis=0)
        var = np.nanvar(base, axis=0)
        
        num_nan = np.sum(np.isnan(base), axis=0)
        base = base - mu
        base[np.isnan(base)] = 0
        stat = np.zeros((self._max_lag, base.shape[1]))
        for lag in range(1, self._max_lag+1):
            stat[lag - 1] = np.sum(base[lag:] * base[:-lag], axis=0) / (base.shape[0] - lag - num_nan)

        stat = stat / var
        self._statistic = stat


    def _set_statistics_boosted(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        # compute the means / var for each stock
        mu = np.nanmean(base, axis=0)
        var = np.nanvar(base, axis=0)
        
        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
        in_matrix = base - mu
        if base.dtype.name == 'float32':
            stat = boosted_stats.lag_prod_mean_float(in_matrix, self._max_lag)
        elif base.dtype.name == 'float64':
            stat = boosted_stats.lag_prod_mean_double(in_matrix, self._max_lag)

        stat /= var
        self._statistic = stat
        # TODO compute outlier if needed
