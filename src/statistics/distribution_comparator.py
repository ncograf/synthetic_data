from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quantile_statistic
import scipy.spatial.distance as sd
import scipy.stats as ss
import statsmodels.api as sm


class DistributionComparator:
    def __init__(
        self,
        real_stat: quantile_statistic,
        syn_stat: quantile_statistic,
        hist_styles: Tuple[Dict[str, any]] = (
            {
                "bins": 100,
                "density": True,
                "color": "blue",
                "label": "Real Data",
                "alpha": 0.5,
            },
            {
                "bins": 100,
                "density": True,
                "color": "green",
                "label": "Synthetic Data",
                "alpha": 0.5,
            },
        ),
        kde_styles: Tuple[Dict[str, any]] = (
            {
                "linestyle": "-",
                "color": "navy",
                "label": "Real KDE",
            },
            {
                "linestyle": "-",
                "color": "darkgreen",
                "label": "Synthetic KDE",
            },
        ),
        ax_style: Dict[str, any] = {
            "title": "No Title",
            "ylabel": r"Density",
            "xlabel": r"Statistic",
            "xscale": "linear",
            "yscale": "linear",
        },
    ):
        self._hist_styles = hist_styles
        self._kde_styles = kde_styles
        self._ax_style = ax_style

        self._stat_one = real_stat
        self._stat_two = syn_stat

        # this only works if statistics are computed before initializeing the comparator

        dat_one = self._stat_one.statistic
        dat_one = dat_one[~np.isnan(dat_one)]
        self._dat_one = dat_one

        dat_two = self._stat_two.statistic
        dat_two = dat_two[~np.isnan(dat_two)]
        self._dat_two = dat_two

        self._divergenes = None

        self._kde_one = sm.nonparametric.KDEUnivariate(dat_one)
        self._kde_one.fit(kernel="epa", fft=False)
        self._kde_two = sm.nonparametric.KDEUnivariate(dat_two)
        self._kde_two.fit(kernel="epa", fft=False)

    def get_divergences(self) -> Tuple[float, float]:
        if self._divergenes is None:
            js_dist = sd.jensenshannon(self._kde_one.density, self._kde_two.density)
            w_dist = ss.wasserstein_distance(self._dat_one, self._dat_two)

            return js_dist, w_dist
        else:
            return self._divergenes

    def draw_distributions(self, ax: plt.Axes):
        ax.set(**self._ax_style)
        ax.plot(self._kde_one.support, self._kde_one.density, **self._kde_styles[0])
        ax.plot(self._kde_two.support, self._kde_two.density, **self._kde_styles[1])
        ax.hist(self._dat_one, **self._hist_styles[0])
        ax.hist(self._dat_two, **self._hist_styles[1])
        js_dist, w_dist = self.get_divergences()
        ax.legend(loc="upper right")
        text_js = f"Jensen-Shannon-Distance {js_dist:.4f}"
        text_w = f"Wasserstein-Distance {w_dist:.4f}"
        text = text_js + "\n" + text_w
        ax.text(
            0.01,
            0.99,
            s=text,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.plot()
