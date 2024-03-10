import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from getch import getch
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
import shutil
import base_statistic
import temporal_statistc
import outlier_detector
from tueplots import bundles

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
        
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams["text.usetex"] = True

    def plot_histogram(self,
                       statistic : base_statistic.BaseStatistic,
                       rc_params : Optional[Dict[str, any]] = None,
                       copy : Optional[Path] = None,
                       density : bool = True):
        """Plots a histogram of the given samples statistic

        Args:
            statistic (base_statistic.BaseStatistic): statistic to be plotted
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            copy (Optional[Path]): path to copy the file to other than the cache directory
            density (bool, optional): Whether to show the density instead of the values. Defaults to True.
        """
        if rc_params is None:
            rc_params = bundles.icml2024()
        
        fig_path = self.cache / f"{statistic._figure_name}_hist.png"

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            statistic.draw_distribution(ax, color="green", density=density)
            fig.savefig(fig_path)
        
        if not copy is None:
            copy_path = copy / f"{statistic._figure_name}_hist.png"
            shutil.copy(fig_path, copy_path)

        plt.show()

    def _check_outliers_manually(self,
                                 statistics : Set[base_statistic.BaseStatistic],
                                 time_statistics : Set[temporal_statistc.TemporalStatistic],
                                 outlier_detectors : Set[outlier_detector.OutlierDetector],
                                 fig : plt.Figure,
                                 ax_time : List[plt.Axes],
                                 ax_stats : List[plt.Axes],
                                 **kwargs
                                 ) -> Dict[str, Set[pd.Timestamp]]:
        
        if len(statistics) != len(ax_stats):
            raise ValueError("For each statistic there must extactly one statistic axis")

        if len(time_statistics) != len(ax_time):
            raise ValueError("For each time statistic there must extactly one time axis")

        outliers = [out_set.get_outliers(**kwargs) for out_set in outlier_detectors]
        all_outliers = set.union(*outliers) # collect all outliers in one set to avoid duplicates
        
        n_outliers = len(all_outliers)
        all_outliers = list(all_outliers)
        all_outliers.sort(key=lambda x : (x[1],x[0])) # sort the outliers inplace
        all_outlier_dates, _ = list(zip(*all_outliers))
        all_outlier_dates = list(all_outlier_dates)

        dict_outlier : Dict[str, Set[pd.Timestamp]]= {}

        for stat, ax in zip(statistics, ax_stats):
            stat.draw_histogram(axes=ax)
        
        for stat, ax in zip(time_statistics, ax_time):
            stat.draw_and_emph_series(ax=ax, dates=all_outlier_dates, neighbour_points=True)

        i : int = 0
        while(i < len(all_outliers)):
            point = all_outliers[i]
            symbol = point[1]
            date = point[0]

            for stat, ax in zip(time_statistics, ax_time):
                stat.cut_window(ax=ax, dates=date, window_size=260)

            for obj, ax in zip(statistics, ax_stats):
                obj.draw_point(axes=ax, point=point)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            print((f"Outlier {i+1}/{n_outliers} of {symbol}: {date}\n"
                  "Press y/n to confirm/reject, h to go back, ctrl-c to cancel, any other key to show next."),flush=True)
            key = getch()

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
                
    def check_outliers_manually(self,
                                statistics : Set[base_statistic.BaseStatistic],
                                time_statistics : Set[temporal_statistc.TemporalStatistic],
                                outlier_detectors : Set[outlier_detector.OutlierDetector],
                                outlier_path : Union[str, Path],
                                **kwargs):

        outlier_path = Path(outlier_path)
        statistics = set(statistics)
        time_statistics = set(time_statistics)
        outlier_detectors = set(outlier_detectors)
        
        all_statistics : Set[base_statistic.BaseStatistic] = statistics.union(time_statistics)

        n_stats = len(statistics)
        n_time_stats = len(time_statistics)
        
        # set ratio of time and stat
        n_col_time = 2
        n_col_stat = 1
        
        n_col = n_col_time
        if  n_stats > 0:
            n_col += n_col_stat # used to get ratio 2:1

        n_rows = np.lcm(n_stats, n_time_stats)
        
        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_col, figure=fig)

        ax_time = []
        for i in range(n_time_stats):
            row_s = i * (n_rows // n_time_stats)
            row_e = (i+1) * (n_rows // n_time_stats)
            ax_time.append(fig.add_subplot(gs[row_s:row_e,:n_col_time]))
            
        ax_stats = []
        for i in range(n_stats):
            row_s = i * (n_rows // n_stats)
            row_e = (i+1) * (n_rows // n_stats)
            ax_stats.append(fig.add_subplot(gs[row_s:row_e,n_col_time:])) # adust ratio here

        plt.ion()
        plt.show()
            
        outlier_dict = {}
        if outlier_path.exists():
            outlier_dict = self._load_outliers(outlier_path)

        for i, col in enumerate(self.data.columns):
        
            for stat in all_statistics:
                stat.set_statistics(data=self.data.loc[:,col])

            # the join of the dictionaries here will be realtively smooth
            print(f"------- CHECK outliers for symbol {col}, {i+1}/{len(self.data.columns)} --------")
            new_outliers = self._check_outliers_manually(statistics=statistics,
                                                         time_statistics=time_statistics,
                                                         outlier_detectors=outlier_detectors,
                                                         fig=fig,
                                                         ax_stats=ax_stats,
                                                         ax_time=ax_time,
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