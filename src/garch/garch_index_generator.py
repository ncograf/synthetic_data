import copy
import multiprocessing as mp
import time
from typing import Dict, List, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd


class GarchIndexGenerator:
    def __init__(self, generator: base_generator.BaseGenerator):
        self._generator = generator
        self._problematic_cols = []

    def fit_and_gen(
        self,
        col_gen_dat: Tuple[str, base_generator.BaseGenerator, pd.Series],
        fit_params: Dict[str, any] = {},
        gen_params: Dict[str, any] = {},
    ) -> Tuple[str, base_generator.BaseGenerator, npt.NDArray, pd.Series]:
        """Fit and generate data for the given input data

        Note that the col element in the tuple is not used but added for conenience when using
        a map function.

        Args:
            col_gen_dat (Tuple[str, base_generator.BaseGenerator, pd.Series]): Tuple with col_name (for convenience, i.e. not used), generator to be fitted, data to fit on.
            fit_params (Dict[str, any], optional): Extra parameters used for the fit function. Defaults to {}.
            gen_params (Dict[str, any], optional): Extra parameters used for the generate function. Defaults to {}.

        Returns:
            Tuple[str, base_generator.BaseGenerator, npt.NDArray, pd.Series]: name, fitted generator, nan_mask, sampled data in the lenght of the input
        """
        col, generator, train_data = col_gen_dat
        mask = ~np.isnan(train_data)
        generator.fit_model(train_data[mask], **fit_params)
        n_samples = train_data[mask].shape[0]
        temp_series, _ = generator.generate_data(len=n_samples, burn=500, **gen_params)
        n = 0
        while np.isinf(temp_series).any() and n < 100:
            temp_series, _ = generator.generate_data(
                len=n_samples, burn=500, **gen_params
            )
            n += 1

        return col, generator, mask, temp_series

    def generate_index(
        self,
        data: pd.DataFrame,
        symbols: List[str] | None = None,
        n_cpu: int = 1,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            data (pd.DataFrame): All data in the index
            symbols (List[str] | None, optional): List of symbols, if None all symbols in the data are considered. Defaults to None
            n_cpu (int): Number of cpu's to use (note the actual number might deviate based on machine limits).
            verbose (bool): Print output.

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame")

        if symbols is None:
            symbols = data.columns

        generated_data = data.copy(deep=True)  # copy to keep the nans
        self._problematic_cols = []

        n_cpu = max(1, min(mp.cpu_count() - 2, n_cpu))
        if n_cpu > 1:
            pool = mp.Pool(n_cpu)

        # create the dictonary and split up
        in_ = [
            (col, copy.deepcopy(self._generator), data.loc[:, col]) for col in symbols
        ]

        before_map = time.time()
        if verbose:
            print("Before Map")
        if n_cpu > 1:
            out = pool.map(self.fit_and_gen, in_)
        else:
            out = map(self.fit_and_gen, in_)
        t = time.time() - before_map
        if verbose:
            print(f"Map took {t:.3f} seconds")

        for col, gen, mask, tmp in out:
            generated_data.loc[mask, col] = tmp

            if np.isinf(tmp).any():
                self._problematic_cols.append(col)

        if verbose:
            print(f"Stocks containing infinity : {self._problematic_cols}.")

        return generated_data.loc[:, symbols]
