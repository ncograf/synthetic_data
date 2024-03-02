import yfinance as yf
import bs4 as bs
import requests
import time
from typing import List, Optional, Union, Dict, Tuple
import pandas as pd
from pathlib import Path
import warnings

class RealData():
    
    def __init__(self, downlaod_delay : int = 1):
        """Utility class to load and transform real data

        Args:
            downlaod_delay (int, optional): Delay between fetching single stocks. Defaults to 1.
        """

        self.download_delay = downlaod_delay
    
    def _get_sp500(self) -> List[str]:
        """Get a list of S&P 500 stock symbols

        The symbols are cleaned to be found on yahoo

        Returns:
            List[str]: List of stock symbols including the index ETF 'SPY'
        """
        # Download stocks names from S&P500 page on wikipedia
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)
        tickers = [s.replace('\n', '') for s in tickers]
        tickers = tickers + ["SPY"]

        # Fix wrong names (BF.B, BRK.B)
        tickers = [s.replace('.', "-") for s in tickers]
        return tickers


    def _get_local_data(self,
                        tickers : List[str],
                        data_path : Union[Path, str] = "data/raw_yahoo_data",
                        required_columns : List[str] = ["Date", "Adj Close"]
                       ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """ Loads local data from files into List of dataFrames
        
        Checks existence and required columns and returnes failed symbols in dict

        Args:
            tickers (List[str]): List of symbols to be checked
            data_path (Union[Path, str], optional): Path to look for files. Defaults to "data/raw_yahoo_data".
            required_columns (List[str], optional): Required columns in each file. Defaults to ["Date", "Adj Close"].

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

        for ticker in tickers:
            
            # check path existence and go to next ticker if failed
            path = data_path / f"{ticker}.csv"
            if (not path.is_file()) or (not path.suffix == ".csv"):
                message = f"For the symbol {ticker} no csv file was found!"
                error_dict[ticker] = message
                continue

            # try to read file and go to next ticker if failed
            try:
                temp_df = pd.read_csv(path)
            except Exception as ex:
                error_dict[ticker] = f"Problem reading csv with pandas for symbol {ticker}:\n{str(ex)}"
                continue
                
            # check existing columns file and go to next ticker if failed
            intersected_cols = set(temp_df.columns).intersection(set(required_columns))
            if not  intersected_cols == set(required_columns):
                message = (f"The file for the symbol {ticker} only contains "
                           f"the columns {intersected_cols} of the required columns {set(required_columns)}.")
                error_dict[ticker] = message
                continue
            
            # check for at least one datapoint
            if temp_df.shape[0] <= 0:
                message = (f"The file for the symbol {ticker} contains not data.")
                error_dict[ticker] = message
                continue
                
            # if all test are completed append to list
            data_dict[ticker] = temp_df

        return data_dict, error_dict        

    def _get_yahoo_data(self, tickers : List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Loads data for all given tickers from Yahoo

        Args:
            tickers (List[str]): List of symbols to be downloaded

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, str]]: symbol -> data, symbol -> error
        """

        data_dict = {}
        error_dict = {}
        
        for i, ticker in enumerate(tickers):
            
            # downloading sigle tickers instead of a list comes down to the
            # same with yfinance library
            try:
                print(f"Downloading symbol {i}: {ticker} from Yahoo.", end=" ")
                data : pd.DataFrame = yf.download(ticker, period="max", interval="1d", progress=False, threads=False)
                print(f"Found {data.shape[0]} datapoints from {data.index[0].strftime('%m-%d-%Y')} to {data.index[-1].strftime('%m-%d-%Y')}.", flush=True)
                time.sleep(self.download_delay)
            except Exception as ex:
                error_dict[ticker] = f"Problem downloading symbol {ticker}:\n{str(ex)}"
                continue

            # check for at least one datapoint
            if data.shape[0] <= 0:
                message = (f"The file for the symbol {ticker} contains not data.")
                error_dict[ticker] = message
                continue

            data_dict[ticker] = data

        return data_dict, error_dict
        

    def _get_all_data(self,
                      data_path : Union[Path, str] = "data/raw_yahoo_data",
                      required_columns : List[str] = ["Date", "Adj Close"],
                      update_all : bool = False
                     ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]:
        """Load all available stock data locally or on yahoo for S&P 500

        Args:
            data_path (Union[Path, str], optional): Local path. Defaults to "data/raw_yahoo_data".
            required_columns (List[str], optional): Columns requiured in the data. Defaults to ["Date", "Adj Close"].
            update_all (bool, optional): If true all data gets loaded from Yahoo. Defaults to False.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]: symbol -> data, symbol -> local_error, symbol -> yahoo_error
        """
        data_path = Path(data_path)
        
        # first get all the symbols
        tickers = self._get_sp500()
        
        # initialize data store
        data_dict = {}
        error_dict = {}

        # get local if update_all is true
        if not update_all:
            data_dict, error_dict = self._get_local_data(tickers=tickers, data_path=data_path, required_columns=required_columns)
            tickers = error_dict.keys()
        
        data_dict_yahoo , error_dict_yahoo = self._get_yahoo_data(tickers=tickers)
        data_dict = data_dict | data_dict_yahoo
        
        # store newly downloaded data
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

        data_dict, local_error, yahoo_error = self._get_all_data(data_path=data_path,
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


test = RealData(5).get_all_data(update_all=True)
print(test.keys())