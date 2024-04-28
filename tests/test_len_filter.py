import len_filter
import numpy as np
import pandas as pd


class TestLenFilter:
    def test_basics(self):
        filter = len_filter.LenFilter(20)
        np_test = np.linspace(1, 200, num=200).reshape((50, 4))
        np_test[:31, 1] = np.nan
        np_test[:30, 0] = np.nan
        pd_test = pd.DataFrame(np_test)
        test = pd_test.copy().to_numpy()
        filter.fit_filter(pd_test)
        filter.apply_filter(pd_test)

        assert np.array_equal(pd_test.to_numpy(), test[:, [0, 2, 3]], equal_nan=True)


if __name__ == "__main__":
    TestLenFilter().test_basics()
