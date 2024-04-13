import pandas as pd
import numpy as np
import numpy.typing as npt
import quantile_statistic as quantile_statistic
import temporal_statistc


class ReturnStatistic(
    quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic
):
    def __init__(self, quantile: float, legend_postfix: str = "", color="green"):
        temporal_statistc.TemporalStatistic.__init__(
            self, legend_postfix=legend_postfix, color=color
        )
        quantile_statistic.QuantileStatistic.__init__(self, quantile)

        self._name = r"Log Returns $R_t = \displaystyle\frac{X_t}{X_{t-1}}$"
        self._sample_name = "S\&P 500 Returns"
        self._figure_name = "sp500_returns"

    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Computes the log returns from the stock prices

        Args:
            data (pd.DataFrame | pd.Series): stock prices
        """
        self.check_data_validity(data)
        self._outlier = None
        self._dates = data.index.to_list()[1:]
        self._symbols = data.columns.to_list()
        self._statistic = self._get_returns(data)
        self._compute_outlier()

    def _get_returns(self, data: pd.DataFrame | pd.Series) -> npt.NDArray:
        """Compute the returns of the data

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

        data[data == 0] = np.nan
        _log_returns = (
            data.iloc[1:, :].to_numpy() / (data.iloc[:-1, :].to_numpy()) + 1e-12
        ) - 1

        return _log_returns
