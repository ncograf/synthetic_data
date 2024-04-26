from typing import List, Tuple

import time
import base_generator
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp


class IndexGenerator:
    def __init__(self, generator: base_generator.BaseGenerator):
        self._generator = generator
        self._problematic_cols = []
        
    def fit_and_gen(self, col_gen_dat : Tuple[str, base_generator.BaseGenerator, pd.Series]):

        col, generator, train_data = col_gen_dat
        mask = ~np.isnan(train_data)
        generator.fit_model(train_data[mask])
        n_samples = train_data[mask].shape[0]
        temp_series, _ = generator.generate_data(len=n_samples, burn=500)
        n = 0
        while np.isinf(temp_series).any() and n < 100:
            temp_series, _ = generator.generate_data(len=n_samples, burn=500)
            n += 1

        return col, generator, mask, temp_series

    def generate_index(
        self, data: pd.DataFrame, symbols: List[str] | None = None, n_cpu : int = 1,
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

        generated_data = data.copy(deep=True) # copy to keep the nans
        self._problematic_cols = []
        
        n_cpu = max(1,min(mp.cpu_count() - 2, n_cpu))
        pool = mp.Pool(n_cpu)
        
        # create the dictonary and split up
        in_ = [(col, copy.deepcopy(self._generator), data.loc[:,col]) for col in symbols]
        

        before_map = time.time()
        print('Before Map')
        out = pool.map(self.fit_and_gen, in_)
        t = time.time() - before_map
        print(f'Map took {t:.3f} seconds')

        for col, gen, mask, tmp in out:
            
            generated_data.loc[mask, col] = tmp

            if np.isinf(tmp).any():
                self._problematic_cols.append(col)

        print(f"Stocks containing infinity : {self._problematic_cols}.")

        return generated_data.loc[:, symbols]
