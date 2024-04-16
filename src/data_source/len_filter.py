import base_filter
import numpy as np
import pandas as pd


class LenFilter(base_filter.BaseFilter):
    def __init__(self, min_points: int = 1000):
        base_filter.BaseFilter.__init__(self)
        self._min_points = min_points
        self.filter_name = "Seqence lenght"

    def fit_filter(self, data: pd.DataFrame | pd.Series):
        """Fit the filter to exclude all data with too little data points

        Args:
            data (pd.DataFrame | pd.Series): data to be filtered
        """

        if isinstance(data, pd.Series):
            data = data.to_frame()

        np_data = data.to_numpy()
        mask = np.isnan(np_data)
        num_not_nan = np.sum(~mask, axis=0)
        mask_columns = num_not_nan >= self._min_points
        self._drop_cols = self._drop_cols.union(
            set(np.array(data.columns)[~mask_columns])
        )
