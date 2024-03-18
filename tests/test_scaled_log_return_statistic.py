import scaled_log_return_statistic as stat
import pandas as pd
import numpy as np
from tick import Tick

class TestScaledLogReturnStatistic:
    
    def test_get_statistic(self):
        
        base_stat = stat.ScaledLogReturnStatistic(quantile=0.01, window=4)
        _symbols = ["a", "b", "c", "d"]
        _log_returns = np.array([2,-2,2,-2,2,-2,2,-2,2,-2])
        n_dates = _log_returns.shape[0] + 1
        _returns = np.exp(_log_returns)
        _prices = np.ones(n_dates)
        for i in range(len(_log_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, 4)), _prices)
        _dates = [pd.Timestamp(f'2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)
        base_stat.set_statistics(df)
        
        assert tuple(base_stat.statistic.shape) == (n_dates - 1, 4)
        assert ((np.abs(base_stat.statistic[2:]) - 1.3483) < 1e-8).all()