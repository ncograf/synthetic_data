import boosted_stats
import pandas as pd
import numpy as np
from tick import Tick
import time

class TestBoosted:

    def test_levearge_effect(self):
        
        rows = 2000
        cols = 500
        max_k = 25
        r_t = np.random.random((rows, cols))
        nan_col = np.random.randint(0,cols, max_k // 3)
        for col in nan_col:
            num_nan = np.random.randint(0,rows-100)
            r_t[:num_nan, col] = np.nan
        r_t_2 = r_t**2

        test = np.zeros((max_k, cols))
        test[:] = np.nan
        for k in range(1, max_k + 1):
            test[k-1,:] = np.nanmean(r_t[:-k] * r_t_2[k:] - r_t[:-k] * r_t_2[:-k], axis=0) / (np.nanmean(r_t[:-k]**2, axis=0)**2)

        boosted = boosted_stats.leverage_effect_double(r_t, max_k)

        assert np.array_equal(test.round(8),boosted.round(8), equal_nan=True)
    
    def test_lag_prod_two(self):

        rows = 2000
        cols = 400
        max_k = 50
        q_t = np.random.random((rows, cols))
        nan_col = np.random.randint(0,cols, max_k // 3)
        for col in nan_col:
            num_nan = np.random.randint(0,rows - 100)
            q_t[:num_nan, col] = np.nan

        r_t = np.random.random((rows, cols))
        nan_col = np.random.randint(0,cols, max_k // 3)
        for col in nan_col:
            num_nan = np.random.randint(0,rows - 100)
            r_t[:num_nan, col] = np.nan
        
        test = np.zeros((max_k, cols))
        test[:] = np.nan

        for k in range(1, max_k + 1):
            test[k-1,:] = np.nanmean(q_t[k:] * r_t[:-k], axis=0)

        boosted = boosted_stats.lag_prod_mean_double(q_t, r_t, max_k)

        assert np.array_equal(test.round(5),boosted.round(5), equal_nan=True)

        for k in range(1, max_k + 1):
            test[k-1,:] = np.nanmean(r_t[k:] * r_t[:-k], axis=0)

        boosted = boosted_stats.lag_prod_mean_double(r_t, max_k)

        assert np.array_equal(test.round(8),boosted.round(8), equal_nan=True)

    def test_gain_loss_asym(self):

        rows = 500
        cols = 20
        max_lag = 500
        theta = 0.1
        q_t = np.random.random((rows, cols))
        nan_col = np.random.randint(0,cols, cols // 3)
        for col in nan_col:
            num_nan = np.random.randint(0,rows - 100)
            q_t[:num_nan, col] = np.nan
        
        t1 = time.time()
        test_loss = np.zeros((max_lag + 1, cols))
        test_gain = np.zeros((max_lag + 1, cols))
        for c in range(cols):
            for t in range(rows):
                for t_ in range(t+1, rows):
                    if t_ - t > max_lag:
                        break
                    if q_t[t_,c] - q_t[t, c] >= theta:
                        test_gain[t_ - t,c] += 1
                        break
                    
        for c in range(cols):
            for t in range(rows):
                for t_ in range(t+1, rows):
                    if t_ - t > max_lag:
                        break
                    if q_t[t_,c] - q_t[t,c] <= -theta:
                        test_loss[t_ - t,c] += 1
                        break

        print("Time for python loops", time.time() - t1, "s")
            
        boosted = boosted_stats.gain_loss_asym_double(q_t, max_lag, theta)
        
        boosted_gain = boosted[0] / boosted[0].sum(axis=0)
        boosted_loss = boosted[1] / boosted[1].sum(axis=0)
        test_gain = test_gain / test_gain.sum(axis=0)
        test_loss = test_loss / test_loss.sum(axis=0)
        
        assert np.array_equal(test_gain.round(8),boosted_gain.round(8), equal_nan=True)
        assert np.array_equal(test_loss.round(8),boosted_loss.round(8), equal_nan=True)
        

if __name__ == '__main__':
    TestBoosted().test_gain_loss_asym()
    #TestBoosted().test_levearge_effect()
    #TestBoosted().test_lag_prod_two()
