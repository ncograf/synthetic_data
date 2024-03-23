import numpy as np
import boosted_stats

class TestBaseStatistic:
    
    def test_simple_float(self):
        
        ones = np.ones((3,4), dtype=np.float32)
        lag_means = boosted_stats.lag_prod_mean_float(ones, 2)
        test = np.ones((2,4), dtype=np.float32)
        
        assert np.array_equal(lag_means, test)

    def test_simple_double(self):
        
        ones = np.ones((3,4), dtype=np.float64)
        lag_means = boosted_stats.lag_prod_mean_double(ones, 2)
        test = np.ones((2,4), dtype=np.float64)
        
        assert np.array_equal(lag_means, test)

    def test_complex_double(self):
        
        n = 3000
        k = 200
        ones = np.ones((n,k), dtype=np.float64)
        m = np.random.randint(0,100, n * k).reshape((n,k))
        arr = np.einsum('kl,kl->kl', ones, m)

        lag = 50
        lag_means = boosted_stats.lag_prod_mean_double(arr, lag)

        test = np.zeros((lag, k), dtype=np.float64)
        for i in range(1,lag+1):
            t = arr[:-i]
            tt = arr[i:]
            test[i-1] = np.sum(t * tt, axis=0) / (n-i)
        
        assert np.array_equal(lag_means, test)


if __name__ == "__main__":
    TestBaseStatistic().test_simple_float()
    TestBaseStatistic().test_simple_double()
    TestBaseStatistic().test_complex_double()