import numpy as np
import numpy.typing as npt
import pandas as pd
import quantile_statistic
import temporal_statistc


class LogReturnStatistic(
    quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic
):
    def __init__(self, quantile: float = 0.01, legend_postfix: str = "", color="green"):
        temporal_statistc.TemporalStatistic.__init__(
            self, legend_postfix=legend_postfix, color=color
        )
        quantile_statistic.QuantileStatistic.__init__(self, quantile)

        self._name = (
            r"Log Returns $R_t = \displaystyle\log\left(\frac{X_t}{X_{t-1}}\right)$"
        )
        self._sample_name = "S\&P 500 Log Returns"
        self._figure_name = "sp500_log_returns"

    def set_statistics(self, data: pd.DataFrame | pd.Series, returns: bool = False):
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
        if returns:
            self._statistic = data.to_numpy()
        else:
            self._statistic = self._get_log_returns(data)
        self._compute_outlier()

    def _get_log_returns(self, data: pd.DataFrame | pd.Series) -> npt.NDArray:
        """Compute the log returns of the data

        Note that this statistic will have one point fewer than the data input

        Returns:
            pd.DataFrame : Log returns
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            pass
        else:
            raise ValueError("Data must be either dataframe or Series")

        data = data.to_numpy()
        _log_returns = np.log(data[1:] / (data[:-1] + 1e-9) + 1e-6)

        return _log_returns
