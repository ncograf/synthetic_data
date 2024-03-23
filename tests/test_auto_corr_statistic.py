import auto_corr_statistic as stat
import log_return_statistic as lstat
import pandas as pd
import numpy as np
from tick import Tick

class TestAutoCorrStatistic:

    def test_get_statistic_simple_boosted(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.AutoCorrStatistic(max_lag=2, underlaying=log_stat, implementation="boosted")

        # prepare data
        _symbols = ["a", "b", "c"]
        _returns = np.exp(np.array([1,2,1,2]))
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, len(_symbols))), _prices)
        _dates = [pd.Timestamp(f'2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)

        # compute statistics
        log_stat.set_statistics(df)
        base_stat.set_statistics()
        assert np.array_equal(base_stat.statistic,np.array([[-1,1],[-1,1],[-1,1]]).T, equal_nan=True)

    def test_get_statistic_simple(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.AutoCorrStatistic(max_lag=2, underlaying=log_stat, implementation='strides')

        # prepare data
        _symbols = ["a", "b", "c"]
        _returns = np.exp(np.array([1,2,1,2]))
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, len(_symbols))), _prices)
        _dates = [pd.Timestamp(f'2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)

        # compute statistics
        log_stat.set_statistics(df)
        base_stat.set_statistics()
        assert np.array_equal(base_stat.statistic,np.array([[-1,1],[-1,1],[-1,1]]).T, equal_nan=True)
    
    def test_get_statistic_python_loop(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.AutoCorrStatistic(max_lag=4, underlaying=log_stat, implementation='python_loop')
        _symbols = ["a", "b", "c", "d"]
        _returns = np.array([2,1,2,1,2,1,2,1,2,1])
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, 4)), _prices)
        _data[:4,1] = np.nan
        _dates = [pd.Timestamp(f'2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)
        log_stat.set_statistics(df)
        base_stat.set_statistics(None)
        assert np.array_equal(base_stat.statistic.round(6),np.array([[-1,1,-1,1],[-1,1,-1,1],[-1,1,-1,1],[-1,1,-1,1]]).T, equal_nan=True)

    def test_get_statistic(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.AutoCorrStatistic(max_lag=4, underlaying=log_stat, implementation='strides')
        _symbols = ["a", "b", "c", "d"]
        _returns = np.array([2,1,2,1,2,1,2,1,2,1])
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, 4)), _prices)
        _data[:4,1] = np.nan
        _dates = [pd.Timestamp(f'2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)
        log_stat.set_statistics(df)
        base_stat.set_statistics(None)
        assert np.array_equal(base_stat.statistic.round(6),np.array([[-1,1,-1,1],[-1,1,-1,1],[-1,1,-1,1],[-1,1,-1,1]]).T, equal_nan=True)

if __name__ == '__main__':
    TestAutoCorrStatistic().test_get_statistic()
    TestAutoCorrStatistic().test_get_statistic_simple()
    TestAutoCorrStatistic().test_get_statistic_python_loop()
