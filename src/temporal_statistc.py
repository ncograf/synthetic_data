import base_statistic
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Optional, List, Set
import pandas as pd
import numpy as np
import numpy.typing as npt


class TemporalStatistic(base_statistic.BaseStatistic):
    """Statistic with additional time series plotting functions"""

    def __init__(self):
        super(TemporalStatistic, self).__init__()
        self.statistic : pd.Series = None # initialize as not but make
    
    def _check_statistics(self):
        """Check whether statistics is valid

        Raises:
            ValueError: If statistics is None, Not a Series, not having Timestamp indices
        """
        
        if self.statistic is None:
            raise ValueError("Statistics must be computed before calling this function.")

        if not isinstance(self.statistic, pd.Series):
            raise ValueError("Statistic must be series")

        if not pd.api.types.is_datetime64_any_dtype(self.statistic.index):
            raise ValueError("Statistic index must be Timestamps")

        if isinstance(self.statistic, pd.Series) and not isinstance(self.statistic.name, str):
            raise ValueError("Statistic name must be string")
        
    
    def _emphasize_points(self, ax : plt.Axes, dates : List[pd.Timestamp], neighbour_points : bool = True):

        self._check_statistics()
        
        if not isinstance(dates, list):
            raise ValueError("Dates must be a list, consider using singleton list.")

        # commented out for performance O(n)
        if not np.isin(dates, self.statistic.index.to_list()).all():
            raise ValueError("Dates were not contained in statistics index")

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

        ax.plot(dates, self.statistic.loc[dates], **style_emph)

        date_loc = np.array([self.statistic.index.get_loc(d) for d in dates])
        
        # plot the neighbours of emphasized plots
        if neighbour_points:
            dates_before = np.maximum(date_loc - 1, 0)
            dates_after = np.minimum(date_loc + 1, self.statistic.shape[0] - 1)
            ax.plot(self.statistic.index[dates_before], self.statistic.iloc[dates_before], **style_neighbours)
            ax.plot(self.statistic.index[dates_after], self.statistic.iloc[dates_after], **style_neighbours)
    
    def _draw_time_series(self, ax : plt.Axes):

        self._check_statistics()

        style_plot = {
            'color' : "green",
            'label' : self.statistic.name,
            'linestyle' : '-',
            'linewidth' : 1,
        }
        
        # plot the investigated symbol
        ax.plot(self.statistic.index, self.statistic, **style_plot)

    def _draw_context(self, ax : plt.Axes, context : pd.DataFrame):

        self._check_data_validity(context)

        style_args = {
            'color' : "gray",
            'alpha' : 0.5,
            'line_width' : 0.8,
            'label' : 'context',
        }

        for col in context.columns:
            ax.plot(context.index, context.loc[:,col], **style_args)
    
    def _plot_labels(self, ax : plt.Axes):
        if not ax.get_legend() is None:
            ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())
    
    def _cut_window(self,
                   ax : plt.Axes,
                   dates : Union[pd.Timestamp, List[pd.Timestamp]],
                   window_size : int
                   ):

        if window_size < 0:
            raise ValueError("Window size was negative.")

        if not isinstance(dates, list):
            dates = [dates]
        
        date_idx = [self.statistic.index.get_loc(date) for date in dates]
        
        fst_date_idx = np.min(date_idx)
        end_date_idx = np.max(date_idx)
        
        fst_idx = np.maximum(fst_date_idx - window_size // 2, 0)
        end_idx = np.minimum(end_date_idx + window_size // 2, self.statistic.shape[0] - 1)
        
        fst_date = self.statistic.index[fst_idx]
        end_date = self.statistic.index[end_idx]

        # adjus the plotted y range to contain all the symbol data and extend the window by context plot
        low_lim = self.statistic.iloc[fst_idx:end_idx].min()
        high_lim = self.statistic.iloc[fst_idx:end_idx].max()
        context_lim = (high_lim - low_lim) * 0.1

        ax.set_ylim((low_lim - context_lim, high_lim + context_lim))
        ax.set_xlim((fst_date, end_date))
    
    def cut_window(self,
                   ax : plt.Axes,
                   dates : Union[pd.Timestamp, List[pd.Timestamp]],
                   window_size : int
                   ):
        """Determines the plotted range using the dates and the window size

        Args:
            ax (plt.Axes): axes to plot
            dates (Union[pd.Timestamp, List[pd.Timestamp]]): date(s) which must be included
            window_size (int): window_size to be plotted
        """
        self._cut_window(ax=ax, dates=dates, window_size=window_size)
        self._plot_labels(ax=ax)

    def draw_and_emph_series(self,
                             ax : plt.Axes,
                             dates : List[pd.Timestamp],
                             neighbour_points : bool = True,
                             ):
        """Draws stock prices the given highliging the given symbol at the given date
        
        The context is chosen randomly if given a integer

        Args:
            ax (plt.Axes): Axes to draw on 
            dates (List[np.datetime64]]): Dates to emphasized extra
            neighbour_points (bool): Whether or not to draw neighbour points next to the critical points
        """
        
        self._draw_time_series(ax)
        self._emphasize_points(ax, dates, neighbour_points=neighbour_points)
        self._plot_labels(ax)
    
    def draw_series(self,
                    ax : plt.Axes):
        """Draw the time series

        Args:
            ax (plt.Axes): axes to plot on
        """
        self._draw_time_series(ax)
        self._plot_labels(ax)
        
    