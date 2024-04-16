from typing import List

import base_filter
import pandas as pd


class ChainedFilter:
    def __init__(self, filter_chain: List[base_filter.BaseFilter]):
        self._drop_cols = []
        self.filter_name = "Chained"
        self.filter_chain = filter_chain

    def fit_filter(self, data: pd.DataFrame | pd.Series):
        """Fit all filters in the filter chain"""

        for filter in self.filter_chain:
            filter.fit_filter(data=data)

    def apply_filter(self, data: pd.DataFrame | pd.Series):
        """Applies all filters in the filter chain

        This method changes the data in place

        Args:
            data (base_statistic.BaseStatistic): statistic to be changed
        """

        for filter in self.filter_chain:
            filter.apply_filter(data)

    def reset_filter(self):
        """Resets the ALL filters in the chain, such that no columns are marked for drop"""

        for filter in self.filter_chain:
            filter.reset_filter()

    def print_state(self, verbose: int = 1):
        """Prints informations about the columns to be dropped with the filter chain

        Args:
            verbosity (int, optional): Determines what information to be displayed
                1 -> only number of droped columns
                2 -> number of droped columns and dropped columns.
                Defaults to 1.
        """

        filter_names = [f"{f.filter_name} filter" for f in self.filter_chain]
        drop_cols = [f._drop_cols for f in self.filter_chain]
        drop_cols = set.union(*drop_cols)

        if verbose >= 1:
            print(
                (
                    f"The chain filter contains the following filters: "
                    f"{filter_names}. All together will will drop {len(drop_cols)} columns."
                )
            )

        if verbose >= 2:
            print((f"Columns marked for drop are: {drop_cols}."))
