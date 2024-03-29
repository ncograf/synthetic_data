import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional, Literal
import powerlaw
from icecream import ic
import matplotlib.pyplot as plt
import temporal_statistc
import scipy.linalg as linalg
import stylized_fact
import boosted_stats

class AutoCorrStatistic(stylized_fact.StylizedFact):
    """Stylized Fact Autocorrelation used for heavy tails and volatility clustering """
    
    def __init__(
            self,
            max_lag : int,
            underlaying : temporal_statistc.TemporalStatistic,
            title : str = 'No Title',
            ylim : Tuple[float] | None = None,
            xscale : str =  'log',
            yscale : str =  'linear',
            powerlaw : bool = False,
            implementation : Literal['boosted', 'strides', 'pyhton_loop'] = 'boosted'):
        
        stylized_fact.StylizedFact.__init__(self)

        self._powerlaw = powerlaw
        self._max_lag = max_lag
        self._underlaying = underlaying
        self._ax_style = {
            'title' : title,
            'ylabel' : r'Auto-correlation',
            'xlabel' : r'lag $k$',
            'xscale' : xscale,
            'yscale' : yscale,
            }
        if not ylim is None:
            self._ax_style['ylim'] = ylim
        self.styles = [{
            'alpha' : 1,
            'marker' : 'o',
            'color' : 'blue',
            'markersize' : 1,
            'linestyle' : 'None',
        }]

        if implementation == 'strides':
            self.set_statistics = self._set_statistics_stride
        elif implementation == 'boosted':
            self.set_statistics = self._set_statistics_boosted
        elif implementation == 'python_loop':
            self.set_statistics = self._set_statistics_python_loop
        else:
            raise ValueError("The given implementation argument does not exist")
        
    def get_alphas(self, xmax=0.05, xmin=0.01):

        self.check_statistic()
        alphas = []
        for c in range(self.statistic.shape[1]):
            dat = np.abs(self.statistic[:,c])
            dat = dat[~np.isnan(dat)]
            fit = powerlaw.Fit(dat, xmax=xmax, xmin=xmin)
            alphas.append(fit.alpha)

        return alphas
    
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

    def draw_stylized_fact(
            self,
            ax : plt.Axes,
            **kwargs
            ):
        """Draws the averaged statistic over all symbols on the axis

        Args:
            ax (plt.Axes): Axis to plot on
        """
        self.check_statistic()
        data = np.mean(self.statistic, axis=1)
        
        ax.set(**self.ax_style)
        if self.x_ticks is None:
            ax.plot(data, **self.styles[0])
        else:
            ax.plot(self.x_ticks, data, **self.styles[0])

        if self._powerlaw:
            alphas = self.get_alphas()
            max_alpha = np.nanmax(alphas)
            min_alpha = np.nanmin(alphas)
            mean_alpha = np.nanmean(alphas)
            text_neg = f"{min_alpha:.4f} " + r'$\leq \alpha \leq$' + f" {max_alpha:.4f}"
            text_pos = r'$\bar{\alpha}$:' + f" {mean_alpha:.4f}"
            text = text_pos + "\n" + text_neg
            ax.text(
                0.01, 0.01,
                s=text,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                )