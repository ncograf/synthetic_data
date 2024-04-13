import pandas as pd
import numpy as np
import numpy.typing as npt
import quantile_statistic
import temporal_statistc


class ScaledLogReturnStatistic(
    quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic
):
    def __init__(
        self, quantile: float, window: int, legend_postfix: str = "", color="green"
    ):
        temporal_statistc.TemporalStatistic.__init__(
            self, legend_postfix=legend_postfix, color=color
        )
        quantile_statistic.QuantileStatistic.__init__(self, quantile)

        self._name = r"Scaled Log Returns $R_t = \displaystyle\log\left(\frac{X_t}{X_{t-1}}\right) / \sigma_{mov}$"
        self._sample_name = r"S\&P 500 Scaled Log Returns"
        self._figure_name = "sp500_scaled_log_returns"
        self._window = window if window % 2 == 1 else window + 1
        self._padding = window // 2

    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Computes the log returns from the stock prices, scaled by a moving std

        Args:
            data (pd.DataFrame | pd.Series): stock prices
        """
        self.check_data_validity(data)
        self._outlier = None
        self._dates = data.index.to_list()[1:]
        self._symbols = data.columns.to_list()
        self._statistic = self._get_scaled_log_returns(data)
        self._compute_outlier()

    def _get_scaled_log_returns(self, data: pd.DataFrame | pd.Series) -> npt.NDArray:
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
        data[data == 0] = np.nan
        _returns = np.log(data[1:, :] / (data[:-1, :] + 1e-12))
        nan_mask = np.isnan(_returns)
        _returns[nan_mask] = 0

        _padded_log_returns = np.pad(
            _returns,
            ((self._padding, self._padding), (0, 0)),  # only pad the rows
            mode="constant",
            constant_values=0,
        )
        _sliding_window_log_retunrs = np.lib.stride_tricks.sliding_window_view(
            _padded_log_returns, self._window, axis=0
        )
        _moving_std = np.std(_sliding_window_log_retunrs, axis=-1)
        _log_returns = _returns / (_moving_std + 1e-12)
        _log_returns[nan_mask] = np.nan

        return _log_returns
