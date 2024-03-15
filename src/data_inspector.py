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
import base_outlier_set
import cached_outlier_set
from tueplots import bundles
import time
from tick import Tick

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
                       symbol : str, 
                       rc_params : Optional[Dict[str, any]] = None,
                       copy : Optional[Path] = None,
                       density : bool = True):
        """Plots a histogram of the given samples statistic

        Args:
            statistic (base_statistic.BaseStatistic): statistic to be plotted
            symbol (str): stock symbol to plot histogram of
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            copy (Optional[Path]): path to copy the file to other than the cache directory
            density (bool, optional): Whether to show the density instead of the values. Defaults to True.
        """
        if rc_params is None:
            rc_params = bundles.icml2024()
        
        fig_path = self.cache / f"{statistic._figure_name}_{symbol}_hist.png"

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            statistic.draw_histogram(ax, symbol=symbol, color="green", y_label="Denstiy" )
            fig.savefig(fig_path)
        
        if not copy is None:
            copy_path = copy / f"{statistic._figure_name}_{symbol}_hist.png"
            shutil.copy(fig_path, copy_path)

        plt.show()

    def _iterate_outliers(self,
                          statistics : Set[base_statistic.BaseStatistic], 
                          time_statistics : Set[temporal_statistc.TemporalStatistic], 
                          outlier_sets : Set[base_outlier_set.BaseOutlierSet], 
                          outlier_path : str | Path, 
                          fig : plt.Figure, 
                          ax_time : List[plt.Axes], 
                          ax_stats : List[plt.Axes], 
                          **kwargs
                          ) -> cached_outlier_set.CachedOutlierSet:
                                 
        if len(statistics) != len(ax_stats):
            raise ValueError("For each statistic there must extactly one statistic axis")

        if len(time_statistics) != len(ax_time):
            raise ValueError("For each time statistic there must extactly one time axis")

        outliers = [out_set.get_outlier() for out_set in outlier_sets]
        all_outliers : Set[Tick] = set.union(*outliers) # collect all outliers in one set to avoid duplicates
        
        outlier_cache = cached_outlier_set.CachedOutlierSet(outlier_path)
        if len(outlier_sets) >= 0:
            outlier_cache.set_outlier(all_outliers)
        
        fig.canvas.flush_events()
        fig.canvas.draw()

        i : int = 0
        last_symbol : str = None
        while(i < len(outlier_cache)):
            point = outlier_cache[i]

            if last_symbol != point.symbol:
                for stat, ax in zip(statistics, ax_stats):
                    stat.draw_histogram(axes=ax, symbol=point.symbol)
                
                for stat, ax in zip(time_statistics, ax_time):
                    stat.restore_time_plot(ax)

                outlier_cache.store_outliers(outlier_path)

            t = time.time()
            for obj, ax in zip(statistics, ax_stats):
                obj.draw_point(axes=ax, point=point)

            for stat, ax in zip(time_statistics, ax_time):
                t = time.time()
                stat.draw_and_emph_series(ax=ax, ticks=point, window_size=200, neighbour_points=True)
            
            fig.canvas.flush_events()
            
            print((f"Outlier {i+1}/{len(outlier_cache)} of {point.symbol}: {point.date}\n"
                  "Press y/n to confirm/reject glitch, h to go back, ctrl-c to cancel, any other key to show next."), flush=True)
            print(time.time() - t, " time all")
            key = getch()

            if key == 'y':
                point.real = False
                print(f"MARKED GLITCH {point.symbol}, {point.date}", flush=True)
                i += 1 # show next element
            elif key == 'n':
                if point.real == False:
                    point.real = True
                    print(f"REMOVED MARK from {point.symbol}: {point.date}", flush=True)
                i += 1 # show next element
            elif key == '\x03':# ctrl-c
                plt.close()
                exit()
            elif key == 'h': # go back
                if(i > 0):
                    i -= 1 # show previous element
                    print(f"Go one back from {point.symbol}: {point.date}", flush=True)
            else:
                i += 1 # show next element

        return outlier_cache
                
    def check_outliers_manually(self,
                                statistics : Set[base_statistic.BaseStatistic],
                                time_statistics : Set[temporal_statistc.TemporalStatistic],
                                outlier_detectors : Set[base_outlier_set.BaseOutlierSet],
                                outlier_path : Union[str, Path],
                                **kwargs):

        outlier_path = Path(outlier_path)
        statistics = set(statistics)
        time_statistics = set(time_statistics)
        outlier_detectors = set(outlier_detectors)
        
        all_statistics : Set[base_statistic.BaseStatistic] = statistics.union(time_statistics)
        pure_outlier_detectors : Set[base_outlier_set.BaseOutlierSet] = outlier_detectors - set.intersection(all_statistics, outlier_detectors)

        n_stats = len(statistics)
        n_time_stats = len(time_statistics)
        
        # set ratio of time and stat
        n_col_time = 2
        n_col_stat = 1
        
        n_col = n_col_time
        if  n_stats > 0:
            n_col += n_col_stat # used to get ratio 2:1

        if n_time_stats <= 0:
            raise ValueError("There must be at least one time statistic for the visual inspection")

        if n_stats > 0:
            n_rows = np.lcm(n_stats, n_time_stats)
        else:
            n_rows = n_time_stats
        
        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_col, figure=fig)

        ax_time = []
        for i in range(n_time_stats):
            row_s = i * (n_rows // n_time_stats)
            row_e = (i+1) * (n_rows // n_time_stats)
            ax = fig.add_subplot(gs[row_s:row_e,:n_col_time])
            ax.set_ylim(auto=True)
            ax.set_xlim(auto=True)
            ax_time.append(ax)

        ax_stats = []
        for i in range(n_stats):
            row_s = i * (n_rows // n_stats)
            row_e = (i+1) * (n_rows // n_stats)
            ax_stats.append(fig.add_subplot(gs[row_s:row_e,n_col_time:])) # adust ratio here

        plt.ion()
        plt.show()
            
        for stat in all_statistics:
            stat.set_statistics(data=self.data)

        # some outlier might depend on the statistic so this needs to be after set statstics
        for out in pure_outlier_detectors:
            out.set_outlier(data=self.data)
            
        fig.axes.clear()

        # the join of the dictionaries here will be realtively smooth
        outlier_cache = self._iterate_outliers(statistics=statistics, 
                                              time_statistics=time_statistics, 
                                              outlier_sets=outlier_detectors, 
                                              outlier_path=outlier_path,
                                              fig=fig, 
                                              ax_stats=ax_stats, 
                                              ax_time=ax_time, 
                                              **kwargs)
        plt.close() 
        
    
    def _view_outliers(self, outlier_dict : Dict[str, List[np.datetime64]]):

        fig, (ax_price, ax_log) = plt.subplots(2,1, sharex=False, sharey=False)
        plt.ion()
        plt.show()
        
        for symbol in outlier_dict.keys():

            self._draw_and_emph_series(symbol=symbol, ax_price=ax_price, ax_log=ax_log, date=outlier_dict[symbol], n_context=0)
            fig.canvas.flush_events()

            print("Press any key to continue.")
            key = getch()
            if key == '\x03':# ctrl-c
                plt.close()
                exit()
            ax_log.clear()
            ax_price.clear()
        
        plt.close()