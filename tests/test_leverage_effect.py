import leverage_effect as stat
import log_return_statistic as lstat
import pandas as pd
import numpy as np

class TestLeverageEffect:

    def test_get_statistic(self):
        
        log_stat = lstat.LogReturnStatistic(0.001)
        base_stat = stat.LeverageEffect(max_lag=4, underlaying=log_stat)
        _symbols = ["a", "b", "c", "d"]
        _returns = np.exp(np.random.rand(10))
        n_dates = _returns.shape[0] + 1
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i+1] = _prices[i] * _returns[i]
        _data = np.einsum('kl,k->kl',np.ones((n_dates, 4)), _prices)
        _data[:4,1] = np.nan
        _dates = [pd.Timestamp('2017-01-01') + pd.Timedelta(days=i) for i in range(n_dates)]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)
        log_stat.set_statistics(df)
        base_stat.set_statistics(None)
        
        enumer = np.zeros((4,4))
        denom = np.zeros((4,4))
        _returns = np.stack([np.log(_returns)]*4, axis=1)
        _returns[:4,1] = np.nan
        count = np.sum(~np.isnan(_returns), axis=0)
        for lag in range(1,5):
            enumer[lag-1] = np.nanmean(_returns[:-lag] * _returns[lag:]**2, axis=0)
            denom[lag-1] = (np.nansum(_returns[lag:]**2, axis=0) / (count - lag))**2 

        test = enumer / denom

        assert np.array_equal(base_stat.statistic.round(6), test.round(6), equal_nan=True)


if __name__ == '__main__':
    TestLeverageEffect().test_get_statistic()
