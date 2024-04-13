import pandas as pd
import numpy as np
import base_filter


class IlliquidityFilter(base_filter.BaseFilter):

    def __init__(self, window : int = 8, min_jumps : int = 2, exclude_tolerance : int = 10):
        self._window = window
        self._min_jumps = min_jumps
        self._data : pd.DataFrame | None = None
        self._exclude_tol : int = exclude_tolerance
        self._drop_cols = []

    def fit_filter(self, data : pd.DataFrame | pd.Series, verbose : bool = True):
        """Filters data which have an illiquid state for strictly longer that self.exclude_tolerance

        A state is considered illiquid if in the given self._window ticks before (including the current)
        the stock did change strictly fewer times than self._min_jumps

        So [1,2,2,3] with a window of 2, and self._min_jumps 1 would only have 1 illiquid state
        So [1,2,2,2,3] with a window of 2, and self._min_jumps 1 would have 2 illiquid state
        
        This method stores the names of columns to be filtered

        Args:
            data (pd.DataFrame | pd.Series): data to filter
        """
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        np_data = data.to_numpy()
        np_diff = np.pad(np_data[:-1] - np_data[1:], pad_width=((1,0),(0,0)))
        mask = np_diff != 0 
        np_diff[mask] = 1
        cum_sum_jumps = np.nancumsum(np_diff, axis=0)
        jump_region_count = cum_sum_jumps[self._window:] - cum_sum_jumps[:-self._window]
        jump_mask = jump_region_count < self._min_jumps
        jumps_sum = np.sum(jump_mask, axis=0)
        no_jump_stock_mask = jumps_sum <= self._exclude_tol

        # simply set all to nan if it does not enough jumps in the region
        self._drop_cols = np.array(data.columns)[~no_jump_stock_mask]
        
        if verbose:
            print("Illiquidity Filter, filteres the following coumns:")
            print(self._drop_cols.tolist())

        