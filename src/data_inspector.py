import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from getch import getch
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Callable, Literal, Set
import base_statistic

class DataInspector:
    """Class to clean data, visualize it and compute basic statistics
    for cleaning and augmentation purposes
    """
    
    def __init__(self, data : pd.DataFrame, cache : str = "data/cache"):
        """Load the data and provide some tools to investigate the data

        Args:
            data (pd.DataFrame): Data with one column for each stock (index needs to be the date)
            cache (str, optional): Directory to find cached files. Defaults to "data/cache".

        Raises:
            ValueError: For bad data input
        """

        if data.shape[0] <= 0:
            raise ValueError("Data inputs not containing anything are not allowed")
    
        if not pd.api.types.is_datetime64_any_dtype(data.index.dtype):
            raise ValueError("Index must be time series data")

        self.data = data
        self.log_returns = self._get_log_returns()
        
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams["text.usetex"] = True

    def draw_log_returns(self, symbols : List[str]):
        """Draw the stock log returns

        Args:
            symbols (List[str]): list of symbols to be plotted
        """
        for col in symbols:
            plt.plot(self.log_returns.index, self.log_returns.loc[:,col])
        
        plt.show()
        
    def draw_stock_price(self, symbols : List[str]):
        """Draw the stock value untransformed

        Args:
            symbols (List[str]): list of symbols to be plotted
        """
        for col in symbols:
            plt.plot(self.data.index, self.data.loc[:,col])
        
        plt.show()

    def _draw_and_emph_series(self,
                            data : pd.DataFrame,
                            ax_series : plt.Axes,
                            symbol : str,
                            dates : Union[np.datetime64, List[np.datetime64]],
                            context : Union[int, List[str]] = 50,
                            neighbour_points : bool = True,
                            window_size : Optional[int] = 200,
                            ):
        """Draws stock prices the given highliging the given symbol at the given date
        
        The context is chosen randomly if given a integer

        Args:
            data (pd.DataFrame): Data to take the plots from
            ax_price (plt.Axes): Axes for the stock price
            symbol (str): symbol to be highlited
            date (Union[np.datetime64, List[np.datetime64]]): dates to emphasized extra
            n_context (Union[int, List[str]], optional):  Additional plotted symbols . Defaults to 50.
            neighbour_points (bool): Whether or not to draw neighbour points next to the critical points
            window_size (Optional[int], optional): size of the window to be shown before and after the date. 
                If None, all dates are plotted. Defaults to 200.
            
        Raises:
            ValueError: n_context needs to be larger-equal 0 and smaller equal number of total symbols if integer
                and window size is requried to be positive.
        """
        
        # check input to be valid
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data was not a DataFrame.")
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index was not a datetime.")

        if not pd.api.types.is_string_dtype(data.columns):
            raise ValueError("Data columns were not string.")
        
        # commented out for performance O(n)
        #if not symbol in data.columns:
        #    raise ValueError("Data did not contain the symbol.")
        
        # commented out for performance O(n)
        #if not dates in data.index:
        #    raise ValueError("Dates were not contained in data.index")
        
        if window_size is None:
            window_size = self.data.shape[0]

        if window_size < 0:
            raise ValueError("Window size was negative.")

        if isinstance(context, int):

            if context < 0 or context > data.shape[1]:
                raise ValueError("Context must either be a list of symbols or .")

            random_choice = np.random.randint(low=0,high=data.shape[1], size=context)
            context = data.columns[random_choice]

        # commented out for performance O(n)
        #elif not context in data.columns:
        #    raise ValueError("Context symbols were not contained in data.columns")

        context_col = "gray"
        context_alpha = 0.5
        context_line_width = 0.8
        
        symbol_col = "green"
        symbol_neigh = "black"
        symbol_line_width=1
        emphasize_col = "red"
        emphasize_size = 5
        
        
        if isinstance(dates, list):
            date_loc = np.array([data.index.get_loc(d) for d in dates])
        else:
            date_loc = np.array([data.index.get_loc(dates)])
        
        symbol_loc = data.columns.get_loc(symbol)
            
        fst = max(date_loc[0] - window_size, 0)
        end = min(date_loc[-1] + window_size,data.shape[0]-1)
        
        
        for col in context:
            ax_series.plot(data.index[fst:end], data.iloc[fst:end][col], c=context_col, alpha=context_alpha, lw=context_line_width, label="context")
        
        # plot the investigated symbol
        ax_series.plot(data.index[fst:end],
                       data.iloc[fst:end,
                       symbol_loc],
                       c=symbol_col,
                       lw=symbol_line_width,
                       label=symbol)

        # plot the neighbours of emphasized plots
        if neighbour_points:
            dates_before = np.maximum(date_loc - 1, 0)
            dates_after = np.minimum(date_loc + 1, data.shape[0])
            ax_series.plot(data.index[dates_before],
                        data.iloc[dates_before,
                        symbol_loc],
                        c=symbol_neigh,
                        marker='o',
                        ms=emphasize_size,
                        label="neighbour point",
                        linestyle='None')
            ax_series.plot(data.index[dates_after],
                           data.iloc[dates_after,
                           symbol_loc],
                           c=symbol_neigh,
                           marker='o',
                           ms=emphasize_size,
                           label="neighbours",
                           linestyle='None')

        ax_series.plot(dates,
                       data.loc[dates, symbol],
                       c=emphasize_col,
                       marker='o',
                       ms=emphasize_size,
                       label="critical point",
                       linestyle='None')
        
        # adjus the plotted y range to contain all the symbol data and extend the window by context plot
        low_lim = data.iloc[fst:end][symbol].min()
        high_lim = data.iloc[fst:end][symbol].max()
        context_lim = (high_lim - low_lim) * 0.1

        ax_series.set(ylim=(low_lim - context_lim, high_lim + context_lim))
        handles, labels = ax_series.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax_series.legend(unique_labels.values(), unique_labels.keys())

    def _check_outliers_manually(self,
                                 statistics : List[base_statistic.BaseStatistic],
                                 fig : plt.Figure,
                                 ax_price : plt.Axes,
                                 ax_log : plt.Axes,
                                 ax_stats : List[plt.Axes],
                                 n_context : int = 50,
                                 **kwargs
                                 ) -> Dict[str, Set[pd.Timestamp]]:
        
        if len(statistics) != len(ax_stats):
            raise ValueError("For each outlier object there must extactly on statistic axis")

        statistic_sets = [obj.get_outliers(**kwargs) for obj in statistics]
        all_outliers = set.union(*statistic_sets)
        
        all_outliers = list(all_outliers)
        all_outliers.sort(key=lambda x : (x[1],x[0]))

        n_outliers = len(all_outliers)
        dict_outlier : Dict[str, Set[pd.Timestamp]]= {}

        for obj, ax, oset in zip(statistics, ax_stats, statistic_sets):
            obj.draw_distribution(axes=ax)

        i : int = 0
        while(i < len(all_outliers)):
            point = all_outliers[i]
            symbol = point[1]
            date = point[0]

            self._draw_and_emph_series(data=self.data, ax_series=ax_price, symbol=symbol, dates=date, context=n_context, window_size=130)
            ax_price.set_label(f"Stock price of {symbol}")
            ax_price.set_xlabel(f"Time daily interval")
            ax_price.set_ylabel(r"Stock Prices $X_t$ in \$")

            self._draw_and_emph_series(data=self.log_returns, ax_series=ax_log, symbol=symbol, dates=date, context=n_context, window_size=130)
            ax_log.set_label(f"Log returns of {symbol}")
            ax_log.set_xlabel(f"Time daily interval")
            ax_log.set_ylabel(r"Log Returns $R_t = \displaystyle\log\left(\frac{X_t}{X_{t-1}}\right)$")

            for obj, ax, oset in zip(statistics, ax_stats, statistic_sets):
                obj.draw_point(axes=ax, index=point, point_array=oset)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print((f"Outlier {i+1}/{n_outliers} of {symbol}: {date}\n"
                  "Press y/n to confirm/reject, h to go back, ctrl-c to cancel, any other key to show next."),flush=True)
            key = getch()
            ax_log.clear()
            ax_price.clear()

            if key == 'y':
                if not (symbol in dict_outlier.keys()):
                    dict_outlier[symbol] = {}
                dict_outlier[symbol].add(date)
                print(f"Added outlier {len(dict_outlier)} {symbol}, {date}", flush=True)
                i += 1 # show next element
            if key == 'n':
                if symbol in dict_outlier.keys() and date in dict_outlier[symbol]:
                    dict_outlier[symbol].remove(date)
                    print(f"Removed outlier from {symbol}: {date}", flush=True)
                i += 1 # show next element
            elif key == '\x03':# ctrl-c
                plt.close()
                exit()
            elif key == 'h': # go back
                if(i > 0):
                    i -= 1 # show previous element
                    print(f"Removed outlier from {symbol}: {date}", flush=True)
            else:
                i += 1 # show next element

        return dict_outlier
    
    def _store_outliers(self, dict_outlier : Dict[str, Set[pd.Timestamp]], path : Union[Path, str]):
        path = Path(path)
        if not path.parent.exists():
            raise FileExistsError("The path to the parent must already exists.")

        for k in dict_outlier:
            temp_ = dict_outlier[k]
            dict_outlier[k] = set([e.floor('d') for e in temp_])

        df = pd.DataFrame.from_dict(dict_outlier, orient="index")
        df.to_json(path, orient='index', date_format='iso')
    
    def _load_outliers(self, path : Union[Path | str]) -> Dict[str, List[pd.Timestamp]]:

        path = Path(path)
        if not path.exists():
            raise FileExistsError(f"The path {str(path)} does not exist.")

        df = pd.read_json(path, orient='index')
        dict_outlier = df.to_dict(orient='index')

        for k in dict_outlier.keys():
            temp_ = dict_outlier[k]
            dict_outlier[k] = set([pd.Timestamp(temp_[e]).floor('d') for e in temp_])

        return dict_outlier 
                
    def check_outliers_manually(self,
                                statistics : Set[base_statistic.BaseStatistic],
                                outlier_path : Union[str, Path],
                                **kwargs):

        outlier_path = Path(outlier_path)
        statistics = set(statistics)
        
        n_stats = len(statistics)
        
        if  n_stats > 0:
            n_rows = ((n_stats + 1) // 2) * 2 # n_rows = roud_to_next_even(len(statistics))
            n_col = 3 # used to get ratio 2:1
        else:
            n_rows = 2
            n_col = 1
        
        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_col, figure=fig)

        if n_stats > 0:
            ax_price = fig.add_subplot(gs[:(n_rows//2),:2]) # adust ratio here
            ax_log = fig.add_subplot(gs[(n_rows//2):,:2])
        else:
            ax_price = fig.add_subplot(gs[0])
            ax_log = fig.add_subplot(gs[1])
            
        ax_stats = []
        for i in range(n_stats):
            ax_stats.append(fig.add_subplot(gs[i,2])) # adust ratio here

        plt.ion()
        plt.show()
            
        outlier_dict = {}
        if outlier_path.exists():
            outlier_dict = self._load_outliers(outlier_path)
        for i, col in enumerate(self.log_returns.columns):
        
            for stat in statistics:
                stat.set_statistics(data=self.data.loc[:,col])

            # the join of the dictionaries here will be realtively smooth
            print(f"------- CHECK outliers for symbol {col}, {i+1}/{len(self.log_returns.columns)} --------")
            new_outliers = self._check_outliers_manually(statistics=statistics,
                                                         fig=fig,
                                                         ax_price=ax_price,
                                                         ax_log=ax_log,
                                                         ax_stats=ax_stats,
                                                         **kwargs)
            outlier_dict = outlier_dict | new_outliers
            self._store_outliers(outlier_dict, outlier_path)
        
        plt.close() 
        
    
    def _view_outliers(self, outlier_dict : Dict[str, List[np.datetime64]]):

        fig, (ax_price, ax_log) = plt.subplots(2,1, sharex=False, sharey=False)
        plt.ion()
        plt.show()
        
        for symbol in outlier_dict.keys():

            self._draw_and_emph_series(symbol=symbol, ax_price=ax_price, ax_log=ax_log, date=outlier_dict[symbol], n_context=0)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print("Press any key to continue.")
            key = getch()
            if key == '\x03':# ctrl-c
                plt.close()
                exit()
            ax_log.clear()
            ax_price.clear()
        
        plt.close()


    def _nan_stats(self, symbols : Optional[List[str]]) -> Dict[str, any]:
        """Print some stats about the data available

        Args:
            symbols (Optional[List[str]]): list of symbols to be taken into account

        Returns:
            Dict[str, any]: Statistics dictonary
        """

        if symbols is None:
            symbols = self.data.columns
            
        data = self.data.loc[:, symbols]
        starts : pd.Series = data.apply(pd.Series.first_valid_index, axis=0)
        def count_nans(col : pd.Series):
            start = starts[col.name]
            return col.loc[start:].isna().sum()
        nan_after_start : pd.Series = data.apply(count_nans)

        def get_last_nan(col : pd.Series):
            if not col.hasnans:
                return col.index[0]
            last_nan_date = col.loc[col.isna()].index.max()
            return col.index[col.index.get_loc(last_nan_date) + 1]
        one_after_last_nan : pd.Series = data.apply(get_last_nan)
        
        print((one_after_last_nan != starts).sum())
        assert ((nan_after_start != 0) == (one_after_last_nan != starts)).all()

        statistics = {
            "Last start date" : one_after_last_nan.max(),
            "Last start date symbol" : one_after_last_nan.index[one_after_last_nan.argmax()],
            "First start date" : one_after_last_nan.min(),
            "Index of last start date" : data.index.get_loc(one_after_last_nan.max()),
            "Total dates" : data.shape[0],
            "Dates on which all are available" : data.shape[0] - data.index.get_loc(one_after_last_nan.max()),
            "Mean start date" : one_after_last_nan.mean(),
            "Mean of nans after start" : nan_after_start.mean(),
        }
        
        for k in statistics.keys():
            print(f"{k:<40}: {statistics[k]}.")
            
        dict_nan_after_start = {}
        for key in nan_after_start.loc[nan_after_start != 0].index:
            dict_nan_after_start[key] = nan_after_start[key]
        
        print(f"Stocks with NAN values after the first valid value:\n", dict_nan_after_start)

            
    def _get_log_returns(self) -> pd.DataFrame:
        """Compute the log returns of the stock prices

        Returns:
            pd.DataFrame : Log returns
        """
        _log_returns = pd.DataFrame(np.log(self.data.iloc[1:,:].to_numpy() / self.data.iloc[:-1,:].to_numpy()))
        _log_returns.index = self.data.index[1:]
        _log_returns.columns = self.data.columns
        return _log_returns