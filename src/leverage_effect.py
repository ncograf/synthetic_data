import pandas as pd
import numpy as np
import numpy.typing as npt
import stylized_fact
import matplotlib as plt
import boosted_stats
import temporal_statistc
from typing import Literal, Dict

class LeverageEffect(stylized_fact.StylizedFact):
    
    def __init__(
            self,
            max_lag : int,
            underlaying : temporal_statistc.TemporalStatistic,
            legend_postfix : str = '',
            color : str = 'blue', 
            ):
        
        stylized_fact.StylizedFact.__init__(self)

        self._name = r"Auto correlation $\displaymath\frac{\mathbb{E}[(r_{t+k} - \mu)(r_t - \mu)]}{\sigma^2}$"
        self._sample_name = underlaying.name
        self._figure_name = r"$L(k)$"
        self._max_lag = max_lag
        self._underlaying = underlaying
        self._plot_color = 'blue'
        self.y_label = r'lag k'

    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        
        mu = np.nanmean(base, axis=0)
        std = np.nanstd(base, axis=0)
        data = base
        
        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
        if base.dtype.name == 'float32':
            stat = boosted_stats.leverage_effect_float(data, self._max_lag)
        elif base.dtype.name == 'float64':
            stat = boosted_stats.leverage_effect_double(data, self._max_lag)

        self._statistic = stat
        # TODO compute outlier if needed