import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional
import quantile_statistic
import temporal_statistc
import scipy.linalg as linalg

class StockPriceStatistic(quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic):
    
    def __init__(self):
        super(quantile_statistic.QuantileStatistic, self).__init__()
        
        self._name = r"Stock Prices $X_t$ in \$"
        self._sample_name = "Stock Price"
        self._figure_name = "sp500_stock_prices"

    def set_statistics(self, data: Union[pd.DataFrame, pd.Series]):
        """Sets the statistic as the data

        Args:
            data (Union[pd.DataFrame, pd.Series]): stock prices
        """
        self._check_data_validity(data)
        self.statistic = data