import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional
import quantile_statistic
import temporal_statistc
import scipy.linalg as linalg

class StockPriceStatistic(quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic):
    
    def __init__(self, quantile : float):
        super(quantile_statistic.QuantileStatistic, self).__init__()

        temporal_statistc.TemporalStatistic.__init__(self)
        quantile_statistic.QuantileStatistic.__init__(self, quantile)
        
        self._name = r"Stock Prices $X_t$ in \$"
        self._sample_name = "Stock Price"
        self._figure_name = "sp500_stock_prices"

    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Sets the statistic as the data

        Args:
            data (pd.DataFrame | pd.Series): stock prices
        """
        self.check_data_validity(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        self._symbols = data.columns.to_list()
        self._dates = data.index.to_list()
        self._statistic = data.to_numpy()