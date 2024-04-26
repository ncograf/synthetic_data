from pathlib import Path
from typing import List, Union

import base_generator
import index_generator
import pandas as pd
import real_data_loader


class GenDataLoader:
    def __init__(self, cache: Union[Path, str] = "data/cache"):
        """Utility class to load and transform real data

        Args:
            downlaod_delay (int, optional): Delay between fetching single stocks. Defaults to 1.
        """

        self.cache: Path = Path(cache)

        # create the cache if it does not yet exist
        self.cache.mkdir(exist_ok=True, parents=True)

    def get_timeseries(
        self,
        generator: base_generator.BaseGenerator,
        data_loader: real_data_loader.RealDataLoader | None = None,
        col_name: str = "Adj Close",
        symbols: List[str] | None = None,
        update_all: bool = False,
        n_cpu : int = 1,
    ) -> pd.DataFrame:
        """Get data, from yahoo or locally depending on whether it is already cached

        Args:
            generator (BaseGenerator) : Generator to generate data
            data_laoder (RealDataLoader | None, optional): Loader to get fitting data if it needs to be created. Defaults to None
            col_name (str): column to be fetched
            symbols (List[str] | None, optional): Symbols to be generated if None all symbols in the data_loader are generated. Defaults to None
            update_all (bool, optional): If true all data gets fetched from yahoo. Defaults to False.
            n_cpu (bool, optional): Number of cpu's to be used to generate the data if needed. Defaults to 1.

        Returns:
            pd.DataFrame : Dataframe containing one column per stock and completed with NAN
        """

        # load cache if possible
        cache_path = (
            self.cache
            / f"time_series_{generator.name}_{str(col_name).replace(' ', '_')}.csv"
        )
        if cache_path.exists() and not update_all:
            try:
                print(f"Cached data found at {cache_path}.")
                data_ = pd.read_csv(cache_path, index_col="Date")
                data_.index = pd.to_datetime(data_.index)
                data_.sort_index(inplace=True)
                if (symbols is None) or (
                    set(symbols).issubset(set(data_.columns.to_list()))
                ):
                    return data_
            except Exception as ex:
                print(f"Loading cache failed. {str(ex)}")

        # both of these functions are well tests, so it hopefully works
        if data_loader is not None:
            price_data = data_loader.get_timeseries(col_name)  # gets data
        else:
            raise ValueError(
                "If data needs to be generated, make sure to pass a data_loader"
            )

        index_gen = index_generator.IndexGenerator(generator=generator)
        data = index_gen.generate_index(price_data, symbols=symbols, n_cpu=n_cpu)
        data.to_csv(cache_path, index=True)
        return data
