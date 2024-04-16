import base_filter
import numpy as np
import pandas as pd


class InftyFilter(base_filter.BaseFilter):
    def __init__(self):
        self._drop_cols = set()
        self.filter_name = "Infinity"

    def fit_filter(self, data: pd.DataFrame | pd.Series):
        """Filters columns (symbols) which have infinity data points

        This method stores the names of columns to be filtered.
        Note that these columns get appended to the already exisitng
        list of columns to be dropped.

        To drop the columns use the `apply_filter` method

        Args:
            data (pd.DataFrame | pd.Series): data to be filter
        """

        if isinstance(data, pd.Series):
            data = data.to_frame()

        no_jump_stock_mask = np.isinf(data.to_numpy()).any(axis=0)

        # simply set all to nan if it does not enough jumps in the region
        new_drop_cols = set(np.array(data.columns)[no_jump_stock_mask].tolist())
        self._drop_cols = new_drop_cols.union(new_drop_cols)
