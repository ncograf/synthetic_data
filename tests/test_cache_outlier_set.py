from tick import Tick
import pandas as pd
from pathlib import Path
import numpy as np
from cached_outlier_set import CachedOutlierSet

class TestCacheOutlierSet:
        
    def test_store_and_load_outliers(self):

        smi_dates = [pd.Timestamp.now(), pd.Timestamp('2017-01-01 12:00:00')]
        spy_dates = [pd.Timestamp.now(), pd.Timestamp('1999-12-18 21:00:00')]
        note = "Note test"
        real = False
        smi_ticks = [Tick(d, "SMI", note, real) for d in smi_dates]
        spy_ticks = [Tick(d, "SPY", note, real) for d in spy_dates]
        tick_set = set(smi_ticks + spy_ticks)

        # make sure the test runs smoothly
        path = "tests/testdata/cache/test_cache.json"
        _set = CachedOutlierSet()
        _set.set_outlier(tick_set)
        _set.store_outliers(path)
        _set.load_outliers(path)
        assert set(_set.outlier) == tick_set
    

    
    
if __name__ == "__main__":
    TestCacheOutlierSet().test_store_and_load_outliers()