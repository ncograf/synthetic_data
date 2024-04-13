import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import temporal_statistc
import stylized_fact
import powerlaw


class NormalizedPriceReturn(stylized_fact.StylizedFact):
    def __init__(
        self,
        underlaying: temporal_statistc.TemporalStatistic,
        title_postfix: str = "",
    ):
        stylized_fact.StylizedFact.__init__(self)
        self._underlaying = underlaying
        self.styles = [
            {
                "alpha": 1,
                "marker": "o",
                "color": "blue",
                "markersize": 1,
                "linestyle": "None",
            },
            {
                "alpha": 1,
                "marker": "o",
                "color": "red",
                "markersize": 1,
                "linestyle": "None",
            },
        ]
        self._ax_style = {
            "title": "heavy-tailed price return" + title_postfix,
            "ylabel": r"$P(r)$",
            "xlabel": r"normalized price return",
            "xscale": "log",
            "yscale": "log",
        }

    def normalized_returns(self) -> npt.NDArray:
        """Compute the empirical 1 - F(X) for the statistic X

        Args:
            symbol (str | None, optional): Symbol to compute it for, computed over all stocks if None. Defaults to None.

        Returns:
            npt.NDArray: Array containing the list of values of X in a first column and (1 - F(X)) in a second column
        """
        self._underlaying.check_statistic()
        nan_mask = np.isnan(self._underlaying.statistic)
        ge_0 = self._underlaying.statistic > 0
        data_pos = self._underlaying.statistic[ge_0 & (~nan_mask)].flatten()
        data_neg = np.abs(self._underlaying.statistic[(~ge_0) & (~nan_mask)].flatten())

        m_pos = data_pos.shape[0]
        m_neg = data_neg.shape[0]
        k = np.min((5000, m_pos, m_neg))
        if m_pos > 0:
            data_pos = np.sort(data_pos)
            std_pos = np.std(data_pos)
            mu_pos = np.mean(data_pos)
            norm_data_pos = (data_pos - mu_pos) / std_pos
            disp_data_pos = norm_data_pos[:: m_pos // k]
            n_pos = disp_data_pos.shape[0]
            P_r_pos = 1 - np.arange(1, n_pos + 1) / n_pos
        else:
            disp_data_pos = np.array([])
            P_r_pos = 1 - np.array([])

        if m_neg > 0:
            data_neg = np.sort(data_neg)
            std_neg = np.std(data_neg)
            mu_neg = np.mean(data_neg)
            norm_data_neg = (data_neg - mu_neg) / std_neg
            disp_data_neg = norm_data_neg[:: m_neg // k]
            n_neg = disp_data_neg.shape[0]
            P_r_neg = 1 - np.arange(1, n_neg + 1) / n_neg
        else:
            disp_data_neg = np.array([])
            P_r_neg = 1 - np.array([])

        n = np.maximum(disp_data_neg.shape[0], disp_data_pos.shape[0])
        out = np.zeros((n, 4))
        out[:] = np.nan
        out[: P_r_neg.shape[0], 0] = P_r_neg
        out[: disp_data_neg.shape[0], 1] = disp_data_neg
        out[: P_r_pos.shape[0], 2] = P_r_pos
        out[: disp_data_pos.shape[0], 3] = disp_data_pos

        return out

    def get_alphas(self, xmin=0.5):
        self.check_statistic()
        dat = np.abs(self.statistic[:, 0])
        dat = dat[np.logical_and(~np.isnan(dat), dat != 0)]
        fit = powerlaw.Fit(dat, xmin=xmin)
        alpha_neg = fit.alpha
        dat = np.abs(self.statistic[:, 2])
        dat = dat[np.logical_and(~np.isnan(dat), dat != 0)]
        fit = powerlaw.Fit(dat, xmin=xmin)
        alpha_pos = fit.alpha
        return alpha_neg, alpha_pos

    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        norm_ret = self.normalized_returns()
        self._statistic = norm_ret

    def draw_stylized_fact(
        self,
        ax: plt.Axes,
    ):
        """Draws the averaged statistic over all symbols on the axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """

        self.check_statistic()

        a_pos, a_neg = self.get_alphas()
        ax.set(**self.ax_style)
        ax.plot(
            self.statistic[:, 3],
            self.statistic[:, 2],
            **self.styles[0],
            label=r"$r_t < 0$",
        )
        ax.plot(
            self.statistic[:, 1],
            self.statistic[:, 0],
            **self.styles[1],
            label=r"$r_t > 0$",
        )
        text_pre = r"$\rho(r) \propto r^{-\alpha}$"
        text_neg = r"$\alpha$ for $(r_t < 0)$: " + f"{a_neg:.4f}"
        text_pos = r"$\alpha$ for $(r_t > 0)$: " + f"{a_pos:.4f}"
        text = text_pre + "\n" + text_pos + "\n" + text_neg
        ax.text(
            0.01,
            0.01,
            s=text,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
        ax.legend()
