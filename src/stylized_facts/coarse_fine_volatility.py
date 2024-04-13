import pandas as pd
import numpy as np
import stylized_fact
import matplotlib.pyplot as plt
import boosted_stats
import temporal_statistc

class CoarseFineVolatility(stylized_fact.StylizedFact):
    
    def __init__(
            self,
            max_lag : int,
            tau : int,
            underlaying : temporal_statistc.TemporalStatistic,
            title_postfix : str = '',
            ):
        
        stylized_fact.StylizedFact.__init__(self)

        self._ax_style = {
            'title' : 'coarse-fine volatility correlation' + title_postfix,
            'ylabel' : r'$\rho(k)$',
            'xlabel' : r'lag $k$',
            'xscale' : 'linear',
            'yscale' : 'linear'
            }
        self._max_lag = max_lag
        self._tau = tau
        self._underlaying = underlaying
        self.styles = [{
            'alpha' : 1,
            'marker' : 'o',
            'color' : 'blue',
            'markersize' : 1,
            'linestyle' : 'None',
        },{
            'alpha' : 1,
            'marker' : 'None',
            'color' : 'orange',
            'markersize' : 1,
            'linestyle' : '-'
        } ]

    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        nan_mask = np.isnan(base)
        base[nan_mask] = 0
        nan_mask_shifted = np.isnan(base[:-self._tau])

        cs_base = np.cumsum(base,axis=0)
        v_c_tau = np.abs(cs_base[self._tau:] - cs_base[:-self._tau])
        v_c_tau[nan_mask_shifted] = np.nan
        v_c_mean = np.nanmean(v_c_tau, axis=0)
        v_c_std = np.nanstd(v_c_tau, axis=0)
        v_c_tau = v_c_tau - v_c_mean

        abs_base = np.abs(base)
        cs_abs_base = np.cumsum(abs_base,axis=0)
        v_f_tau = cs_abs_base[self._tau:] - cs_abs_base[:-self._tau]
        v_f_tau[nan_mask_shifted] = np.nan
        v_f_mean = np.nanmean(v_f_tau, axis=0)
        v_f_std = np.nanstd(v_f_tau, axis=0)
        v_f_tau = v_f_tau - v_f_mean
        
        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
        if base.dtype.name == 'float32':
            stat_pos = boosted_stats.lag_prod_mean_float(v_c_tau, v_f_tau, self._max_lag)
            stat_neg = boosted_stats.lag_prod_mean_float(v_f_tau, v_c_tau, self._max_lag)
        elif base.dtype.name == 'float64':
            stat_pos = boosted_stats.lag_prod_mean_double(v_c_tau, v_f_tau, self._max_lag)
            stat_neg = boosted_stats.lag_prod_mean_double(v_f_tau, v_c_tau, self._max_lag)

        stat_neg = stat_neg / (v_c_std * v_f_std)
        stat_pos = stat_pos / (v_c_std * v_f_std)
        stat = np.concatenate([np.flip(stat_neg, axis=0), stat_pos], axis=0)
        ticks = np.arange(1,self._max_lag + 1)
        ticks = np.concatenate([-np.flip(ticks), ticks])
        self._x_ticks = ticks
        self._statistic = stat
        
        self._lead_lag = self.statistic[:self._max_lag] - self.statistic[self._max_lag:]
        # TODO compute outlier if needed

    def draw_stylized_fact(
            self,
            ax : plt.Axes,
            ):
        """Draws the averaged statistic over all symbols on the axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """
        
        self.check_statistic()
        data = np.mean(self.statistic, axis=1)
        lead_log = np.mean(self._lead_lag, axis=1)
        lead_log_x = np.arange(1,self._max_lag + 1)
        
        ax.set(**self.ax_style)
        ax.plot(self.x_ticks, data, **self.styles[0])
        ax.plot(lead_log_x, lead_log, **self.styles[1])
