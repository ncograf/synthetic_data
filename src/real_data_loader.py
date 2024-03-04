import yfinance as yf
import bs4 as bs
import requests
import time
from typing import List, Union, Dict, Tuple
import pandas as pd
from pathlib import Path

class RealDataLoader():
    
    def __init__(self, download_delay : int = 1, cache : Union[Path, str] = "data/cache"):
        """Utility class to load and transform real data

        Args:
            downlaod_delay (int, optional): Delay between fetching single stocks. Defaults to 1.
        """

        self.download_delay : int = download_delay
        self.cache : Path = Path(cache)
        
        # create the cache if it does not yet exist
        self.cache.mkdir(exist_ok=True,parents=True)

    
    def _get_snp_500(self,
                     data_path : Union[Path, str] = "data/raw_yahoo_data",
                     update_all : bool = False,
                     ) -> List[str]:
        """Get a list of S&P 500 stock symbols


        
        Args:
            data_path (Union[Path, str], optional): Path to look for files. Defaults to "data/raw_yahoo_data".
            update_all (bool): Indicates wheter to refetch the list from Yahoo

        Returns:
            List[str]: List of stock symbols including the index ETF 'SPY'
        """
        
        symbol_path = Path(data_path) / "symbols.csv"
        
        if not update_all and symbol_path.exists() and symbol_path.is_file():
            symbols = list(pd.read_csv(symbol_path))
        else:
            # Download stocks names from S&P500 page on wikipedia
            resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            symbols = []
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                symbols.append(ticker)
            symbols = [s.replace('\n', '') for s in symbols]
            symbols = symbols + ["SPY"]

            # Fix wrong names (BF.B, BRK.B)
            symbols = [s.replace('.', "-") for s in symbols]
            
            pd.Series(symbols).to_csv(symbol_path, index=False)

        return symbols


    def _get_local_data(self,
                        symbols : List[str],
                        data_path : Union[Path, str] = "data/raw_yahoo_data",
                        required_columns : List[str] = ["Adj Close"]
                       ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """ Loads local data from files into List of dataFrames
        
        The column 'Date' is always requried and will be used as index
        
        Checks existence and required columns and returnes failed symbols in dict

        Args:
            symbols (List[str]): List of symbols to be checked
            data_path (Union[Path, str], optional): Path to look for files. Defaults to "data/raw_yahoo_data".
            required_columns (List[str], optional): Required columns in each file. The column 'Date' is always required. Defaults to ["Adj Close"].

        Raises:
            FileExistsError: If the given data_path does not exist

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, str]]: Mapping symbol -> stock data, Mapping symbol -> error
        """

        data_path = Path(data_path)
        if not data_path.exists() or not data_path.is_dir():
            message = f"The given folder {str(data_path.absolute())} does not exist or is not a directory!"
            raise FileExistsError(message)
        
        # initialize output dicts
        data_dict = {}
        error_dict = {}

        for symbol in symbols:
            
            # check path existence and go to next ticker if failed
            path = data_path / f"{symbol}.csv"
            if (not path.is_file()) or (not path.suffix == ".csv"):
                message = f"For the symbol {symbol} no csv file was found!"
                error_dict[symbol] = message
                continue

            # try to read file and go to next ticker if failed
            try:
                temp_df = pd.read_csv(path)
            except Exception as ex:
                error_dict[symbol] = f"Problem reading csv with pandas for symbol {symbol}:\n{str(ex)}"
                continue
                
            # check existing columns file and go to next ticker if failed
            required_columns = required_columns + ['Date'] # add Date because it is always required
            intersected_cols = set(temp_df.columns).intersection(set(required_columns))
            if not  intersected_cols == set(required_columns):
                message = (f"The file for the symbol {symbol} only contains "
                           f"the columns {intersected_cols} of the required columns {set(required_columns)}.")
                error_dict[symbol] = message
                continue
            
            # check for at least one datapoint
            if temp_df.shape[0] <= 0:
                message = (f"The file for the symbol {symbol} contains not data.")
                error_dict[symbol] = message
                continue
            
            # set index
            temp_df = temp_df.set_index('Date')
                
            # if all test are completed append to list
            data_dict[symbol] = temp_df

        return data_dict, error_dict        

    def _get_yahoo_data(self, symbols : List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Loads data for all given tickers from Yahoo
        
        Args:
            symbols (List[str]): List of symbols to be downloaded

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, str]]: symbol -> data, symbol -> error
        """

        data_dict = {}
        error_dict = {}
        
        for i, symbol in enumerate(symbols):
            
            # downloading sigle tickers instead of a list comes down to the
            # same with yfinance library
            try:
                print(f"Downloading symbol {i}: {symbol} from Yahoo.", end=" ")
                data : pd.DataFrame = yf.download(symbol, period="max", interval="1d", progress=False, threads=False)
                print(f"Found {data.shape[0]} datapoints from {data.index[0].strftime('%m-%d-%Y')} to {data.index[-1].strftime('%m-%d-%Y')}.", flush=True)
                time.sleep(self.download_delay)
            except Exception as ex:
                error_dict[symbol] = f"Problem downloading symbol {symbol}:\n{str(ex)}"
                continue

            # check for at least one datapoint
            if data.shape[0] <= 0:
                message = (f"The file for the symbol {symbol} contains not data.")
                error_dict[symbol] = message
                continue
            
            data_dict[symbol] = data

        return data_dict, error_dict
        

    def _get_all_data(self,
                      symbols : List[str],
                      data_path : Union[Path, str] = "data/raw_yahoo_data",
                      required_columns : List[str] = ["Date", "Adj Close"],
                      update_all : bool = False,
                      store_downloads : bool = True
                     ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]:
        """Load all available stock data locally or on yahoo for S&P 500

        Args:
            symbols (List[str]): List of symbols to be downloaded
            data_path (Union[Path, str], optional): Local path. Defaults to "data/raw_yahoo_data".
            required_columns (List[str], optional): Columns requiured in the data. Defaults to ["Date", "Adj Close"].
            update_all (bool, optional): If true all data gets loaded from Yahoo. Defaults to False.
            store_downloads (bool, optional): If true the data fetched from Yahoo is locally stored. Defaults to True.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]: symbol -> data, symbol -> local_error, symbol -> yahoo_error
        """
        data_path = Path(data_path)
        
        # initialize data store
        data_dict = {}
        error_dict = {}

        # get local if update_all is true
        if not update_all:
            data_dict, error_dict = self._get_local_data(symbols=symbols, data_path=data_path, required_columns=required_columns)
            symbols = error_dict.keys()
        
        data_dict_yahoo , error_dict_yahoo = self._get_yahoo_data(symbols=symbols)
        data_dict = data_dict | data_dict_yahoo
        
        # store newly downloaded data
        if store_downloads:
            for key in data_dict_yahoo.keys():
                data_dict_yahoo[key].to_csv(data_path / f"{key}.csv")
            
        return data_dict, error_dict, error_dict_yahoo
    

    def get_all_data(self,
                     data_path : Union[Path, str] = "data/raw_yahoo_data",
                     required_columns : List[str] = ["Date", "Adj Close"],
                     update_all : bool = False
                     ) -> Dict[str, pd.DataFrame]:
        """Load all S&P 500 data into a dictonary of dataframes

        Args:
            data_path (Union[Path, str], optional): Local datapath. Defaults to "data/raw_yahoo_data".
            required_columns (List[str], optional): Columns that need to be present in the data. Defaults to ["Date", "Adj Close"].
            update_all (bool, optional): If true all data gets fetched from yahoo. Defaults to False.

        Returns:
            Dict[str, pd.DataFrame]: Dictonary with all available symbols and corresponding data
        """
        # first get all the symbols
        symbols = self._get_snp_500(data_path=data_path, update_all=update_all)
        data_dict, local_error, yahoo_error = self._get_all_data(
                                                                 symbols=symbols,
                                                                 data_path=data_path,
                                                                 required_columns=required_columns,
                                                                 update_all=update_all)
        
        # print errors
        if len(local_error) > 0:
            for err in local_error.keys():
                print(f"Local Error: {err}")

        # yahoo errors
        if len(yahoo_error) > 0:
            for err in yahoo_error.keys():
                print(f"Yahoo Error: {err}")
        
        return data_dict

    def _merge_columns(self,
                       column_name : str,
                       data_dict : Dict[str, pd.DataFrame],
                       ) -> pd.DataFrame:
        """Extract the given  from the data dict and merge them into one dataframe
        
        Converts the index column into datetime and sorts by that

        Args:
            column_name (str): Column to be merged
            data_dict (Dict[str, pd.DataFrame]): Dictionary to be merged together

        Returns:
            pd.DataFrame: Dataframe with merged columns
        """
        selection_list = [ data_dict[k].loc[:,column_name] for k in data_dict.keys() ]
        data_ = pd.concat(selection_list, axis=1, join='outer')
        data_.columns = data_dict.keys()
        data_.index = pd.to_datetime(data_.index)
        data_.sort_index(inplace=True)
        return data_

    
    def get_timeseries(self,
                      col_name : str = "Adj Close",
                      data_path : Union[Path, str] = "data/raw_yahoo_data",
                      update_all : bool = False,
                     ) -> pd.DataFrame:
        """Get data, from yahoo or locally depending on whether it is already cached

        Args:
            col_name (str): column to be fetched
            data_path (Union[Path, str], optional): Local data Path to laod data from. Defaults to "data/raw_yahoo_data".
            update_all (bool, optional): If true all data gets fetched from yahoo. Defaults to False.

        Returns:
            pd.DataFrame : Dataframe containing one column per stock and completed with NAN
        """

        # load cache if possible
        cache_path = self.cache / f"time_series_{str(col_name).replace(' ', '_')}.csv" 
        if cache_path.exists() and not update_all:
            try:
                print(f"Cached data found at {cache_path}.")
                data_ = pd.read_csv(cache_path, index_col="Date")
                data_.index = pd.to_datetime(data_.index)
                data_.sort_index(inplace=True)
                return data_
            except Exception as ex:
                print(f"Loading cache failed. {str(ex)}")

        # both of these functions are well tests, so it hopefully works
        data_dict = self.get_all_data(data_path=data_path, required_columns=["Date", col_name], update_all=update_all)
        data_ =  self._merge_columns(column_name=col_name, data_dict=data_dict)
        data_.to_csv(cache_path)
        return data_
