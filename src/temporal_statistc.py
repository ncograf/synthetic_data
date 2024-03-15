import base_statistic
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Optional, List, Set
import pandas as pd
import numpy as np
import numpy.typing as npt
from tick import Tick


class TemporalStatistic(base_statistic.BaseStatistic):
    """Statistic with additional time series plotting functions"""

    def __init__(self):
        super(TemporalStatistic, self).__init__()
        self._statistic : npt.NDArray | None = None # initialize as not but make
        self.series_artists = {}
        
    def restore_time_plot(self, ax : plt.Axes):
        """Clears the axis and the artist series

        Args:
            ax (plt.Axes): axes to be cleaned
        """
        ax.clear()
        self.series_artists = {}
        
    def check_statistics(self):
        """Check whether statistics is valid

        Raises:
            ValueError: If statistics is None, Not a Series, not having Timestamp indices
        """
        
        if self.statistic is None:
            raise ValueError("Statistics must be computed before calling this function.")

        if not isinstance(self.statistic, np.ndarray):
            raise ValueError("Statistic must be numpy array")

        if not len(self.dates) == self.statistic.shape[0]:
            raise ValueError("Statistics must have same fist axis as dates")

        if not len(self.symbols) == self.statistic.shape[1]:
            raise ValueError("Second axis of statistics must be of the same size as symbols")
        
    def _emphasize_ticks(self, ax : plt.Axes,
                         ticks : List[Tick],
                         neighbour_points : bool = True):

        self.check_statistics()
        
        if not self.statistic.ndim == 2:
            raise RuntimeError("When emphasizing points, statistic needs to be one dimensional")
        
        if not isinstance(ticks, list):
            raise ValueError("Dates must be a list, consider using singleton list.")

        style_emph = {
            'color' : "red",
            'marker' : 'o',
            'markersize' : 5,
            'label' : 'critical points',
            'linestyle' : 'None',
        }

        style_neighbours = {
            'color' : "black",
            'marker' : 'o',
            'markersize' : 3,
            'label' : 'neighbour points',
            'linestyle' : 'None',
        }

        if not 'emph_points' in self.series_artists.keys():
            self.series_artists['emph_points'] = ax.plot([], [], **style_emph)[0]
        
        time_data = np.array([tick.date for tick in ticks])
        date_loc = np.array([self.get_date_index(tick) for tick in ticks])
        symbol_loc = np.array([self.get_symbol_index(tick) for tick in ticks])
        stat_data = np.array(self.statistic[date_loc, symbol_loc]) 

        self.series_artists['emph_points'].set_xdata(time_data)
        self.series_artists['emph_points'].set_ydata(stat_data)

        # plot the neighbours of emphasized plots
        if neighbour_points:
            if not 'neigh_before' in self.series_artists.keys():
                self.series_artists['neigh_before'] = ax.plot([],[], **style_neighbours)[0]
            if not 'neigh_after' in self.series_artists.keys():
                self.series_artists['neigh_after'] = ax.plot([],[], **style_neighbours)[0]

            dates_before = np.maximum(date_loc - 1, 0)
            dates_after = np.minimum(date_loc + 1, self.statistic.shape[0] - 1)

            self.series_artists['neigh_before'].set_xdata(np.array(self.dates)[dates_before])
            self.series_artists['neigh_before'].set_ydata(self.statistic[dates_before, symbol_loc])

            self.series_artists['neigh_after'].set_xdata(np.array(self.dates)[dates_after])
            self.series_artists['neigh_after'].set_ydata(self.statistic[dates_after, symbol_loc])
    

    def _draw_time_series(self,
                          ax : plt.Axes,
                          symbol : str,
                          fst_date : pd.Timestamp,
                          end_date : pd.Timestamp,
                          ):

        # no checks for performance O(n)
        # self._check_statistics()

        style_plot = {
            'color' : "green",
            'label' : symbol,
            'linestyle' : '-',
            'linewidth' : 1,
        }
        
        # plot the investigated symbol
        fst_idx = self.get_date_index(fst_date)
        end_idx = self.get_date_index(end_date)
        symbol_idx = self.get_symbol_index(symbol)
        time_region = self.dates[fst_idx:end_idx]
        statistic_region = self.statistic[fst_idx:end_idx, symbol_idx]
        if not f'time_series_{symbol}' in self.series_artists.keys():
            self.series_artists[f'time_series_{symbol}'] = ax.plot([],[],**style_plot)[0]
        
        self.series_artists[f'time_series_{symbol}'].set_data((time_region, statistic_region))

    def _draw_context(self,
                      ax : plt.Axes,
                      context : List[str],
                      fst_date : pd.Timestamp,
                      end_date : pd.Timestamp,
                      ):

        # no checks for performance
        # self._check_data_validity(context)

        style_args = {
            'color' : "gray",
            'alpha' : 0.5,
            'line_width' : 0.8,
            'label' : 'context',
        }

        fst_idx = self.get_date_index(fst_date)
        end_idx = self.get_date_index(end_date)
        time_region = self.dates[fst_idx:end_idx]
        for symbol in context:
            # TODO clear all symbols that are not used any more
            symbol_idx = self.get_symbol_index(symbol)
            if not f'context_{symbol}' in self.series_artists.keys():
                self.series_artists[f'context_{symbol}'] = ax.plot([],[],**style_args)[0]
            
            self.series_artists[f'context_{symbol}'].set_xdata(time_region)
            self.series_artists[f'context_{symbol}'].set_ydata(self.statistic[fst_idx:end_idx, symbol_idx])
    
    def _plot_labels(self, ax : plt.Axes):
        
        # plot only unique labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))

        if not 'time_legend' in self.series_artists.keys():
            # fix position for performance
            self.series_artists['time_legend'] = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')
    
    def _get_end_points(self,
                        ticks : Tick | List[Tick],
                        window_size : int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        
        if window_size < 0:
            raise ValueError("Window size was negative.")

        if not isinstance(ticks, list):
            ticks = [ticks]
        
        date_idx = [self.get_date_index(tick) for tick in ticks]
        
        fst_date_idx = np.min(date_idx)
        end_date_idx = np.max(date_idx)
        
        fst_idx = np.maximum(fst_date_idx - window_size // 2, 0)
        end_idx = np.minimum(end_date_idx + window_size // 2, self.statistic.shape[0] - 1)
        
        fst_date = self.dates[fst_idx]
        end_date = self.dates[end_idx]

        return (fst_date, end_date)
    
    def _set_limits(self,
                    ax : plt.Axes,
                    symbols : List[str],
                    fst_date : pd.Timestamp,
                    end_date : pd.Timestamp,
                    ):

        fst_idx = self.get_date_index(fst_date)
        end_idx = self.get_date_index(end_date)
        sym_idxs = [self.get_symbol_index(symbol) for symbol in symbols]

        # adjus the plotted y range to contain all the symbol data and extend the window by context plot
        low_lim = np.nanmin(self.statistic[fst_idx:end_idx, sym_idxs])
        high_lim = np.nanmax(self.statistic[fst_idx:end_idx, sym_idxs])
        context_lim = (high_lim - low_lim) * 0.1

        ax.set_ylim((low_lim - context_lim, high_lim + context_lim), emit=False)
        ax.set_xlim((fst_date, end_date), emit=False)
    
    def _draw_labels(self, ax : plt.Axes):
        """Computes the labels and adds them to the axes

        Args:
            ax (plt.Axes): axes to add labels
        """
        ax.set_xlabel("Dates")
        ax.set_ylabel(self._name)
    
    def draw_and_emph_series(self,
                             ax : plt.Axes,
                             ticks : Tick | List[Tick],
                             window_size : int,
                             neighbour_points : bool = True,
                             ):
        """Draws stock prices the given highliging the given symbol at the given date
        
        The context is chosen randomly if given a integer

        Args:
            ax (plt.Axes): Axes to draw on 
            dates (List[np.datetime64]]): Dates to emphasized extra
            window_size (int): Indicates the size of the shown data points
            neighbour_points (bool): Whether or not to draw neighbour points next to the critical points
        """
        if not isinstance(ticks, list):
            ticks = [ticks]

        symbols = set([tick.symbol for tick in ticks]) 

        fst_date, end_date = self._get_end_points(ticks, window_size=window_size)
        self._set_limits(ax=ax, symbols=symbols, fst_date=fst_date, end_date=end_date)
        for symbol in symbols:
            self._draw_time_series(ax, symbol, fst_date=fst_date, end_date=end_date)
        self._emphasize_ticks(ax, ticks=ticks, neighbour_points=neighbour_points)
        self._plot_labels(ax)
        self._draw_labels(ax)
    
    def draw_series(self,
                    ax : plt.Axes):
        """Draw the time series

        Args:
            ax (plt.Axes): axes to plot on
        """
        fst_date = pd.Series.first_valid_index(self.statistic)
        self._draw_time_series(ax, fst_date, end_date=self.statistic.index[-1])
        self._plot_labels(ax)
        
    