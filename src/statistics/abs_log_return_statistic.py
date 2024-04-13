import log_return_statistic
import numpy as np
import pandas as pd


class AbsLogReturnStatistic(log_return_statistic.LogReturnStatistic):
    def __init__(self, quantile: float, legend_postfix: str = "", color="green"):
        log_return_statistic.LogReturnStatistic.__init__(
            self, quantile=quantile, legend_postfix=legend_postfix, color=color
        )

        self._name = r"Abs Log Returns $|R_t| = \left|\displaystyle\log\left(\frac{X_t}{X_{t-1}}\right)\right|$"
        self._sample_name = "S\&P 500 Abs Log Returns"
        self._figure_name = "sp500_abs_log_returns"

    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Computes the log returns from the stock prices

        Args:
            data (pd.DataFrame | pd.Series): stock prices
        """
        self.check_data_validity(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        self._outlier = None
        self._dates = data.index.to_list()[1:]
        self._symbols = data.columns.to_list()
        self._statistic = np.abs(self._get_log_returns(data))
        self._compute_outlier()  # TODO fix one sided
