import pandas as pd
import numpy as np
from typing import Dict
import base_statistic

class SP500Statistic(base_statistic.BaseStatistic):
    
    def __init__(self):
        super(base_statistic.BaseStatistic, self).__init__()
        
        self._name = "Number of days"
        self._sample_name = "S\&P 500 Stocks"
        self._figure_name = "sp500_stocks"
        self._plot_color = 'green'

    def _check_data_validity(self, data : pd.DataFrame):
        """Check whether the data is a dataframe and has the right types

        Data must be:
            - DataFrame
            - Time in index
            - String in column

        Args:
            data (pd.DataFrame): data to be checked

        Raises:
            ValueError: If the specifications are not met
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be dataframe.")

        if not pd.api.types.is_string_dtype(data.columns):
            raise ValueError("Data column names must be string.")

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be Timestamps")

    def set_statistics(self, data: pd.DataFrame):
        """Computes the number of datapoints per stock

        Args:
            data (Union[pd.DataFrame, pd.Series]): data
        """
        self._check_data_validity(data)
        self._statistic = data.notna().sum(axis=0) # compute number of valid datapoints in each symbol
        self._statistic = self.statistic.T.to_numpy().reshape((-1,1))
        self._symbols = ['num_data']
        self.start_series : pd.Series = data.apply(pd.Series.first_valid_index, axis=0)

        def _start_clean_series(col : pd.Series):
            """Computes the first date after which (and including) only non nan values appear"""
            if not col.hasnans:
                return col.index[0]
            last_nan_date = col.loc[col.isna()].index.max()
            last_idx = np.minimum(col.index.get_loc(last_nan_date) + 1, len(col) - 1)
            return col.index[last_idx]
        self.start_clean_series : pd.Series = data.apply(_start_clean_series)

        def _nan_after_start(col : pd.Series):
            """Counts the number of nan elements after the first valid element"""
            start = self.start_series[col.name]
            return col.loc[start:].isna().sum()
        self.nan_after_start : pd.Series = data.apply(_nan_after_start)
        
        self.dict_messy_series = {}
        for key in data.columns[self.nan_after_start != 0]:
            self.dict_messy_series[key] = [self.nan_after_start[key], self.start_series[key], self.start_clean_series[key]]
        
        self.total_dates = data.shape[0]
        
    def distribution_properties(self) -> Dict[str, any]:
        """Comptue properties of distribution

        Raises:
            ValueError: If statistic is not available

        Returns:
            Dict[str, any]: distribution properties
        """

        if self.statistic is None:
            raise ValueError("Statistic must be set before calling this function")
        
        properties = {
            "Number of symbols" : self.statistic.shape[0],
            "Number of symbols avove 1000 points": np.sum(self.statistic >= 1000),
            "Last clean start date" : self.start_clean_series.max(),
            "Last clean start date symbol" : self.start_clean_series.index[self.start_clean_series.argmax()],
            "Last start date" : self.start_series.max(),
            "Last start date symbol" : self.start_series.index[self.start_series.argmax()],
            "First clean start date" : self.start_clean_series.min(),
            "First start date" : self.start_series.min(),
            "Total dates" : self.total_dates,
            "Mean clean start date" : self.start_clean_series.mean(),
            "Mean data points" : self.statistic.mean(),
            "Max data points" : self.statistic.max(),
            "Min data Points" : self.statistic.min(),
            "Median data Points" : np.median(self.statistic),
            "Stocks with NAN values after the first valid value:\n" : self.dict_messy_series
        }
        
        return properties
        

    def print_distribution_properties(self):
        """Print the properies of the distribution"""

        properties = self.distribution_properties()
        for k in properties.keys():
            if not isinstance(properties[k], dict):
                print(f"{k:<40} {properties[k]}.")

        for k in properties.keys():
            if isinstance(properties[k], dict):
                print(f"{k}{properties[k]}")