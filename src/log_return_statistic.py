import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional
import quantile_statistic
import temporal_statistc
import scipy.linalg as linalg

class LogReturnStatistic(quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic):
    
    def __init__(self):
        super(LogReturnStatistic, self).__init__()
        self._name = r"Log Returns $R_t = \displaystyle\log\left(\frac{X_t}{X_{t-1}}\right)$"
        self._sample_name = "S\&P 500 Log Returns"
        self._figure_name = "sp500_log_returns"

    def set_statistics(self, data: Union[pd.DataFrame, pd.Series]):
        """Computes the log returns from the stock prices

        Args:
            data (Union[pd.DataFrame, pd.Series]): stock prices
        """
        self._check_data_validity(data)
        self.statistic = self._get_log_returns(data)

    def _get_log_returns(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Compute the log returns of the data

        Note that this statistic will have one point fewer than the data input

        Returns:
            pd.DataFrame : Log returns
        """
        if isinstance(data, pd.Series):
            _log_returns = pd.Series(np.log(data.iloc[1:].to_numpy() / data.iloc[:-1].to_numpy()))
            _log_returns.name = data.name
        elif isinstance(data, pd.DataFrame):
            _log_returns = pd.DataFrame(np.log(data.iloc[1:,:].to_numpy() / data.iloc[:-1,:].to_numpy()))
            _log_returns.columns = data.columns
        else:
            raise ValueError("Data must be either dataframe or Series")

        _log_returns.index = data.index[1:]
        return _log_returns