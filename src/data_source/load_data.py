import os
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance as yf


def _load_prices(symbols: List[str], path: Path) -> pd.DataFrame:
    """Downloads symbols from yahoo finance if not locally cached in directory path

    Args:
        symbols (list): Symbols to downlaod
        path (Path): Directory to cache the stock data

    Returns:
        dataframe: Price data one stock per column
    """

    # create path to cache data
    path.mkdir(exist_ok=True, parents=True)
    index_path = path / "all_index.csv"

    if index_path.exists():
        temp_df = pd.read_csv(index_path, index_col="Date")
        if set(symbols) <= set(temp_df.columns):
            return temp_df.loc[:, symbols]

    # load data from disk or yahoo finance
    data_dict = {}
    for symbol in symbols:
        symbol_path = path / f"{symbol}.csv"

        if not symbol_path.exists():
            # download stock from yahoo finance
            print(f"Downloading {symbol} from Yahoo.")
            data: pd.DataFrame = yf.download(
                symbol, period="max", interval="1d", progress=False, threads=False
            )
            data.to_csv(symbol_path)
            data_dict[symbol] = data
        else:
            # laod stock from disk
            temp_df = pd.read_csv(symbol_path)
            temp_df = temp_df.set_index("Date")
            data_dict[symbol] = temp_df

    # merge index into one dataframe
    COL_NAME = "Adj Close"
    selection_list = []
    for k in data_dict.keys():
        temp_data: pd.Series = data_dict[k].loc[:, COL_NAME]
        temp_data = temp_data.dropna()
        selection_list.append(temp_data)

    selection_list = [data_dict[k].loc[:, COL_NAME] for k in data_dict.keys()]
    data = pd.concat(selection_list, axis=1, join="outer")
    data.columns = data_dict.keys()
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # store for later use
    data.to_csv(index_path)

    return data


def get_log_returns(
    price_data: pd.DataFrame, min_len: int, verbose: bool = False
) -> Tuple[npt.NDArray, pd.Index]:
    """Compute log return data and filter data based for sufficient data

    Args:
        price_data (pd.DataFrame): price data
        min_len (int): minimal number of prices listed to pass filter
        verbose (bool, optional): print information of data

    Returns:
        npt.NDArray, list: log returns for stocks (T x #stocks), stock_symbols
    """

    symbols = price_data.columns

    if verbose:
        print(f"Timespan {price_data.index[-min_len]} -- {price_data.index[-1]}")

    price_data = np.asarray(price_data, dtype=np.float64)
    nan_mask = ~np.isnan(price_data)  # returns pd.dataframe
    num_non_nans = np.sum(nan_mask, axis=0)

    log_return_data = np.log(price_data[1:] / price_data[:-1])
    log_return_data = log_return_data[-min_len:, num_non_nans >= min_len]
    log_return_data[np.isnan(log_return_data)] = 0

    symbols = list(symbols[num_non_nans >= min_len])

    if verbose:
        print(f"Number of symbols {len(symbols)}")

    return log_return_data, symbols


def load_prices(index: Literal["dax", "smi", "sp500"]) -> pd.DataFrame:
    """Load prices from yahoo or locally from disk if avalable

    Args:
        index ("dax" | "smi" | "sp500"): Index to be loaded

    Returns:
        pd.DataFrame: price dataframe for given index
    """

    index_path = Path(__file__).parent / f"{index}_symbols.csv"
    symbols = pd.read_csv(index_path, header=None).iloc[:, 0].to_list()
    path = Path(os.environ["DATA_DIR"]) / f"{index}_raw_data"

    return _load_prices(symbols, path)


def load_log_returns(
    index: Literal["dax", "smi", "sp500"],
    min_len: int = 4096,
    symbols: List[str] = [],
) -> npt.NDArray:
    """loads log return data for the given index and
    filters all symbols with less than min_len datapoints

    Args:
        index ("dax" | "smi" | "sp500"): Index to be loaded
        min_len (int, optional): Minimial required datapoints for filtering. Defaults to 4096.
        symobls (list): List of symbols to be considered, if emptly all symobls are considered. Defautls to [].

    Returns:
        npt.NDArray: log returns for the index (min_len x #stocks)
    """

    prices = load_prices(index)
    if len(symbols) > 0:
        prices = prices.loc[:, symbols]

    log_ret, _ = get_log_returns(prices, min_len)
    return log_ret
