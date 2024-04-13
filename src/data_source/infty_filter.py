import pandas as pd
import numpy as np
import base_filter


class InftyFilter(base_filter.BaseFilter):

    def __init__(self):
        self._drop_cols = []

    def fit_filter(self, data : pd.DataFrame | pd.Series, verbose : bool = True):
        """Filters data which have inifity in the data

        This method stores the names of columns to be filtered

        Args:
            data (pd.DataFrame | pd.Series): data to filter
        """
        
        if isinstance(data, pd.Series):
            data = data.to_frame()

        no_jump_stock_mask = np.isinf(data.to_numpy()).any(axis=0)

        # simply set all to nan if it does not enough jumps in the region
        self._drop_cols = np.array(data.columns)[no_jump_stock_mask]
        
        if verbose:
            print("Infinity Filter, filteres the following coumns:")
            print(self._drop_cols.tolist())
        

        