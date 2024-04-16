import pandas as pd


class BaseFilter:
    def __init__(self):
        self._drop_cols = set()
        self.filter_name = "Base"

    def fit_filter(self, data: pd.DataFrame | pd.Series):
        raise NotImplementedError("The data filter needs to be impleemnted first")

    def apply_filter(self, data: pd.DataFrame | pd.Series):
        """Applies the filter to the given statistic, i.e. removes the columns which are unnecessary

        Args:
            stat (base_statistic.BaseStatistic): statistic to be changed
        """

        symbols = []
        for sym in self._drop_cols:
            if sym in data.columns.to_list():
                symbols.append(sym)

        data.drop(symbols, axis=1, inplace=True)

    def reset_filter(self):
        """Resets the filter, such that no columns are marked for drop"""
        self._drop_cols = set()

    def print_state(self, verbose: int = 1):
        """Prints informations about the columns to be droped

        Args:
            verbosity (int, optional): Determines what information to be displayed
                1 -> only number of droped columns
                2 -> number of droped columns and dropped columns.
                Defaults to 1.
        """

        if verbose >= 1:
            print(
                (
                    f"{self.filter_name} Filter, will drop {len(self._drop_cols)} columns."
                )
            )

        if verbose >= 2:
            print((f"Columns marked for drop are: {self._drop_cols}."))
