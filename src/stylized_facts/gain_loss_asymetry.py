import pandas as pd
import numpy as np
import numpy.typing as npt
import stylized_fact
import matplotlib.pyplot as plt
import boosted_stats
import temporal_statistc
from typing import Literal, Dict

class GainLossAsymetry(stylized_fact.StylizedFact):
    
    def __init__(
            self,
            max_lag : int,
            theta : float,
            underlying_price : temporal_statistc.TemporalStatistic,
            ):
        
        stylized_fact.StylizedFact.__init__(self)

        self._ax_style = {
            'title' : 'gain/loss asymetry',
            'ylabel' : r'return time probability',
            'xlabel' : r"time ticks t'",
            'xscale' : 'log',
            'yscale' : 'linear'
            }
        self.styles = [{
            'alpha' : 1,
            'marker' : 'o',
            'color' : 'red',
            'markersize' : 1,
            'linestyle' : 'None',
        },{
            'alpha' : 1,
            'marker' : 'o',
            'color' : 'blue',
            'markersize' : 1,
            'linestyle' : 'None',
        }]
        self._max_lag = max_lag
        self._underlaying = underlying_price
        self._theta = theta

    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately

        log_price = np.log(base)
        n = log_price.shape[0]
        if log_price.dtype.name == 'float32':
            boosted = boosted_stats.gain_loss_asym_float(log_price, self._max_lag, self._theta, True)
        if log_price.dtype.name == 'float64':
            boosted = boosted_stats.gain_loss_asym_double(log_price, self._max_lag, self._theta, True)
        else:
            raise ValueError(f"Unsupported data type: {log_price.dtype.name}")
        boosted_gain = boosted[0] / boosted[0].sum(axis=0)
        boosted_loss = boosted[1] / boosted[1].sum(axis=0)
        self._statistic = (boosted_gain, boosted_loss)
        # TODO compute outlier if needed

    def check_statistic(self):
        """Ensures that statistic is set and at least two dimensional

        Raises:
            ValueError: Raises error if statistic is not set
        """
        if self._statistic is None:
            raise ValueError("Statistics must be computed before being referenced.")

        if not isinstance(self._statistic, tuple):
            raise ValueError("Statistic must be a tuple.")

    def draw_stylized_fact(
            self,
            ax : plt.Axes,
            ):
        """Draws the averaged statistic over all symbols on the axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """

        self.check_statistic()
        
        ax.set(**self.ax_style)
        mean_gain = np.nanmean(self.statistic[0],axis=1)
        mean_loss = np.nanmean(self.statistic[1],axis=1)
        ax.plot(mean_gain, **self.styles[0], label=r'$\theta > 0$')
        ax.plot(mean_loss, **self.styles[1], label=r'$\theta < 0$')
        ax.legend()