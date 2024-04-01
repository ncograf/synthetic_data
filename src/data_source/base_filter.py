import pandas as pd
import numpy as np
import base_statistic
from icecream import ic

class BaseFilter:

    def __init__(self):
        self._drop_cols = []

    def fit_filter(self, data : pd.DataFrame | pd.Series) -> pd.DataFrame:
        raise NotImplementedError("The data filter needs to be impleemnted first")
        
    def apply_filter(self, data : pd.DataFrame | pd.Series):
        """Applies the filter to the given statistic, i.e. removes the columns which are unnecessary

        Args:
            stat (base_statistic.BaseStatistic): statistic to be changed
        """
        
        indices = []
        for sym in self._drop_cols:
            idx = data.columns.to_list().index(sym)
            indices.append(idx)

        data.drop(self._drop_cols, axis=1, inplace=True)