from typing import List

import base_generator
import numpy as np
import pandas as pd


class IndexGenerator:
    def __init__(self, generator: base_generator.BaseGenerator):
        self._generator = generator
        self._problematic_cols = []

    def generate_index(
        self, data: pd.DataFrame, symbols: List[str] | None = None
    ) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            data (pd.DataFrame): All data in the index
            symbols (List[str] | None, optional): List of symbols, if None all symbols in the data are considered. Defaults to None

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame")

        if symbols is None:
            symbols = data.columns

        generated_data = data.copy(deep=True)
        self._problematic_cols = []

        for col in symbols:
            train_data = data.loc[:, col]
            mask = ~np.isnan(train_data)
            self._generator.fit_model(train_data[mask])
            n_samples = train_data[mask].shape[0]
            temp_series, _ = self._generator.generate_data(len=n_samples, burn=500)
            n = 0
            while np.isinf(temp_series).any() and n < 100:
                temp_series, _ = self._generator.generate_data(len=n_samples, burn=500)
                n += 1

            if n == 100:
                self._problematic_cols.append(col)

            generated_data.loc[mask, col] = temp_series

        print(f"Stocks containing infinity : {self._problematic_cols}.")

        return generated_data.loc[:, symbols]
