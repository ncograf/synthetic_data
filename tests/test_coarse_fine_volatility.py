import coarse_fine_volatility as stat
import log_return_statistic as lstat
import pandas as pd
import numpy as np

class TestCoarseFineVolatility:

    def test_get_statistic_simple_boosted(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.CoarseFineVolatility(max_lag=2, underlaying=log_stat, tau=1)

        # prepare data
        _symbols = ["a", "b", "c"]
        _returns = np.exp(np.array([1,2,1,2,1]))
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, len(_symbols))), _prices)
        _dates = [pd.Timestamp('2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)

        # compute statistics
        log_stat.set_statistics(df)
        base_stat.set_statistics()
        assert np.array_equal(base_stat.statistic[:2,:].round(8),-np.array([[-1,1],[-1,1],[-1,1]]).round(8).T, equal_nan=True)
        assert np.array_equal(base_stat.statistic[2:,:].round(8),np.array([[-1,1],[-1,1],[-1,1]]).round(8).T, equal_nan=True)

if __name__ == '__main__':
    TestCoarseFineVolatility().test_get_statistic_simple_boosted()
