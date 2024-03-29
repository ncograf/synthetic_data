import pandas as pd
import numpy as np
from icecream import ic

class BaseFilter:

    def __init__(self):
        self._data : pd.DataFrame | None = None

    def filter_data(self, data : pd.DataFrame | pd.Series) -> pd.DataFrame:
        raise NotImplementedError("The data filter needs to be impleemnted first")
        

    def get_data(self, data : pd.DataFrame | pd.Series | None) -> pd.DataFrame:
        """Returns the filtered data
        
        If data was computed before return that data, otherwise filter the input

        Args:
            data (pd.DataFrame | pd.Series | None): data to be filtered, only used if self._data = None

        Raises:
            RuntimeError: If the data is not provided and the filtered data was not computed before

        Returns:
            pd.DataFrame: filtered data
        """
        
        if data is None and not self._data is None:
            return self._data

        if not data is None:
            return self.filter_data(data)
    
        raise RuntimeError("No data was found or provided.")



