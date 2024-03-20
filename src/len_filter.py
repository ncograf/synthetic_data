import pandas as pd
import numpy as np
import base_filter
from icecream import ic


class LenFilter(base_filter.BaseFilter):

    def __init__(self, min_points : int = 1000):
        base_filter.BaseFilter.__init__(self)
        self._min_points = min_points

    def filter_data(self, data : pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Filters all stocks and returns only the ones with sufficient data points

        Args:
            data (pd.DataFrame | pd.Series): data to be filtered

        Returns:
            pd.DataFrame : dataframe containing the filtered data
        """
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        np_data = data.to_numpy()
        mask = np.isnan(np_data)
        num_not_nan = np.sum(~mask, axis=0)
        mask_columns = num_not_nan >= self._min_points
        self._data = pd.DataFrame(np_data[:,mask_columns], columns=data.columns[mask_columns], index=data.index)
        
        return self._data