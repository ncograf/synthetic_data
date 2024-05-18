import base_filter
import pandas as pd


class TimeFilter(base_filter.BaseFilter):
    def __init__(self, first_date: pd.Timestamp):
        self._first_date = first_date
        self._data: pd.DataFrame | None = None
        self._drop_cols = set()
        self.filter_name = "Time"

    def fit_filter(self, data: pd.DataFrame | pd.Series):
        """Nothing to do here"""
        pass

    def apply_filter(self, data: pd.DataFrame | pd.Series):
        """Filters rows before self._first_date

        This method stores the names of columns to be filtered

        Args:
            data (pd.DataFrame | pd.Series): data to filter
        """

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be Timestamps")

        mask = data.index >= self._first_date
        data.drop(data.index[~mask], inplace=True)
