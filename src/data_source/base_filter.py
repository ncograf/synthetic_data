import pandas as pd

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
        
        symbols = []
        for sym in self._drop_cols:
            if sym in data.columns.to_list():
                symbols.append(sym)

        data.drop(symbols, axis=1, inplace=True)