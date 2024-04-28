import boosted_stats
import numpy as np
import pandas as pd
import stylized_fact
import temporal_statistc


class LeverageEffect(stylized_fact.StylizedFact):
    def __init__(
        self,
        max_lag: int,
        underlaying: temporal_statistc.TemporalStatistic,
        title_postfix: str = "",
    ):
        stylized_fact.StylizedFact.__init__(self)

        self._ax_style = {
            "title": "leverage effect" + title_postfix,
            "ylabel": r"$L(k)$",
            "xlabel": r"lag $k$",
            "xscale": "linear",
            "yscale": "linear",
        }
        self.styles = [
            {
                "alpha": 1,
                "marker": "None",
                "color": "blue",
                "markersize": 1,
                "linestyle": "-",
            }
        ]
        self._sample_name = underlaying.name
        self._max_lag = max_lag
        self._underlaying = underlaying

    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        # get statistic usually log returns
        self._underlaying.check_statistic()
        base = self._underlaying.statistic
        self._symbols = self._underlaying.symbols
        # std = np.nanstd(base, axis=0)
        mu = np.nanmean(base, axis=0)
        data = base - mu

        # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
        if base.dtype.name == "float32":
            stat = boosted_stats.leverage_effect_float(data, self._max_lag, False)
        elif base.dtype.name == "float64":
            stat = boosted_stats.leverage_effect_double(data, self._max_lag, False)

        self._statistic = stat
        # TODO compute outlier if needed
