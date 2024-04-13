import pandas as pd
import numpy as np
import illiquidity_filter


class TestIlliquidityFilter:
    def test_advanced(self):
        filter = illiquidity_filter.IlliquidityFilter(
            window=3, min_jumps=2, exclude_tolerance=7
        )
        n = 200
        pd_test = pd.DataFrame(np.linspace(1, n, num=n).reshape((50, n // 50)))
        pd_test.iloc[:10, 0] = 0  # 9 jumps
        pd_test.iloc[3:7, 3] = 0
        pd_test.iloc[3:8, 1] = 0  # 4 jumps
        pd_test.iloc[13:17, 1] = 0  # 3 jumps
        pd_test.iloc[3:8, 2] = 0  # 5 jumps
        pd_test.iloc[13:18, 2] = 0  # 5 jumps
        filter.fit_filter(pd_test)
        test = pd_test.copy().to_numpy()
        filter.apply_filter(pd_test)

        t = pd_test.to_numpy() == test[:, [1, 3]]
        assert np.all(t)

    def test_basics(self):
        filter = illiquidity_filter.IlliquidityFilter(
            window=3, min_jumps=2, exclude_tolerance=7
        )
        pd_test = pd.DataFrame(np.linspace(1, 100, num=100).reshape((50, 2)))
        pd_test.iloc[:10, 0] = 0
        filter.fit_filter(pd_test)

        test = pd_test.copy().to_numpy()
        filter.apply_filter(pd_test)

        t = pd_test.to_numpy() == test[:, [1]]
        assert np.all(t)


if __name__ == "__main__":
    TestIlliquidityFilter().test_advanced()
