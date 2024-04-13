import scaled_log_return_statistic as stat
import pandas as pd
import numpy as np


class TestScaledLogReturnStatistic:
    def test_get_statistic(self):
        base_stat = stat.ScaledLogReturnStatistic(quantile=0.01, window=2)
        _symbols = ["a", "b", "c", "d"]
        _returns = np.array([2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
        n_dates = _returns.shape[0] + 1
        _returns = _returns
        _prices = np.ones(n_dates)
        for i in range(len(_returns)):
            _prices[i + 1] = _prices[i] * _returns[i]
        _data = np.einsum("kl,k->kl", np.ones((n_dates, 4)), _prices)
        _dates = [
            pd.Timestamp("2017-01-01") + pd.Timedelta(days=i) for i in range(n_dates)
        ]
        df = pd.DataFrame(_data, columns=_symbols, index=_dates)
        base_stat.set_statistics(df)

        assert tuple(base_stat.statistic.shape) == (n_dates - 1, 4)
        local_var_2 = (np.log(2) - (np.log(1) + np.log(1) + np.log(2)) / 3) ** 2
        local_var_1 = (np.log(1) - (np.log(1) + np.log(1) + np.log(2)) / 3) ** 2
        test = np.log(1) / np.sqrt((local_var_1 + local_var_1 + local_var_2) / 3)
        test_t = test
        local_var_2 = (np.log(2) - (np.log(2) + np.log(1) + np.log(2)) / 3) ** 2
        local_var_1 = (np.log(1) - (np.log(2) + np.log(1) + np.log(2)) / 3) ** 2
        test = np.log(2) / np.sqrt((local_var_2 + local_var_1 + local_var_2) / 3)
        assert (np.abs(base_stat.statistic[[3, 5, 7]] - test_t) < 1e-8).all()
        assert (np.abs(base_stat.statistic[[2, 4, 6]] - test) < 1e-8).all()


if __name__ == "__main__":
    TestScaledLogReturnStatistic().test_get_statistic()
