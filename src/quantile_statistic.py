import base_statistic
import base_outlier_set
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend import Legend
from typing import Union, Tuple, Set, Callable, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt
from tick import Tick

class QuantileStatistic(base_statistic.BaseStatistic, base_outlier_set.BaseOutlierSet):
    """Statistic with additional qualitative and quantitative measures
    to find outliers.
    """
    
    def __init__(self, quantile : float):
        """Quantile Statistic collecting outliers

        Args:
            quantile (float): One sided quantile to be cosidered as outlier
            metric (Callable[[npt.ArrayLike], float] | None, optional): 
                Metric used on higher dimensional statistic to determine outliers. Defaults to np.array.

        Raises:
            ValueError: _description_
        """

        base_statistic.BaseStatistic.__init__(self)
        base_outlier_set.BaseOutlierSet.__init__(self)

        self.point_artist : Line2D = None
        self.histogram_legend_artist : Legend = None
        self.quantile = quantile
        
        # check argument
        if self.quantile <= 0 or self.quantile >= 1:
            raise ValueError("The quantile must be within 0 and 1")
    
    @property
    def data(self) -> npt.ArrayLike:
        return self.statistic

    def _get_index(self, mask : npt.ArrayLike) -> Set[Tick]:
        """ Compute a list of indices and columns from mask

        Args:
            mask (npt.ArrayLike): mask of which to get the indices

        Returns:
            List[Tuple[pd.Timestamp, str]]: List with indices where the mask is true
        """
        
        self.check_statistic()

        np_indices = np.where(mask)
        if len(np_indices) == 0:
            return set()
        dates = np.array(self.dates)[np_indices[0]]
        symbols = np.array(self.symbols)[np_indices[1]]

        return Tick.zip(dates=dates, symbols=symbols)
        
    def _get_mask(self) -> npt.NDArray:
        """Compute outliers based on the quanitle set by the class

        Returns:
            Union[List[Tuple[str, np.datetime64]], List[str]]: List containing the index (and column depending on the data)of outliers
        """
        
        self.check_statistic()

        # compute thesholds
        assert self.statistic.ndim == 2
        thresh_low = np.nanpercentile(self.statistic, self.quantile / 2 * 100, axis=0)
        thresh_high = np.nanpercentile(self.statistic, (1 - self.quantile / 2) * 100, axis=0)
        
        mask = (self.statistic > thresh_high) | (self.statistic < thresh_low)
        return mask
    
    def get_outlier(self) -> Set[Tick]:
        """Computes a list of outlier points in the dataframe / series

        If dataframe a list of tuples is returned, otherwise a list of indices

        Returns:
            List[Tuple[pd.Timestamp, str]]: List with indices where the mask is true
        """
        self.check_outlier()
        return self._outlier

    def _compute_outlier(self):
        """Computes the outliers based on the given statistics, qunatile and metric"""
        mask = self._get_mask()
        self._outlier = self._get_index(mask=mask)
        
        
    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Sets the statistic to given data

        Args:
            data (pd.DataFrame | pd.Series): statistic samples
        """
        self.check_data_validity(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        self._dates = data.index.to_list()
        self._symbols = data.columns.to_list()
        self._statistic = data.to_numpy()
        self._compute_outlier()
        
    
    def set_outlier(self, data: pd.Series | pd.DataFrame):
        """Alias for set statistic

        Args:
            data (pd.Series | pd.DataFrame): data to set the statistic
        """
        self.set_statistics(data)
        
    def draw_point(self,
                   axes : plt.Axes,
                   point: Tick,
                   outlier_style = {
                      "color" : "red",  
                      "alpha" : 1,
                   },
                   normal_style = {
                      "color" : "green",  
                      "alpha" : 1,
                   },
                   ):
        """Draws a single datapoint as a vertical line

        The plot can e.g. be the histogram plot in which it makes sense

        Args:
            axes (plt.Axes): Plot to draw the line into
            point (Tick): datapoint (date, symbol, ...) to be plotted
            symbol (str): Symbol to be plotted
            outlier_style (Dict[str, any]): Keyword arguments to plot funciton for points
            normal_style (Dict[str, any]): Keyword arguments to plot funciton for points

        Raises:
            NotImplementedError: Only (time)Series data is supported
        """
                   
        self.check_statistic()

        # get bin containing the index
        if self.point_artist is None:
            self.point_artist = axes.axvline()

        x = self.get_statistic(point)
        self.point_artist.set_xdata(x)
        self.point_artist.set_label(f"{point.symbol} at {point.date.strftime('%Y-%m-%d')}")

        if self.is_outlier(tick=point):
            self.point_artist.set(**outlier_style)
        else:
            self.point_artist.set(**normal_style)

        handles, labels = axes.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        axes.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    def draw_histogram(self,
                       axes : plt.Axes,
                       symbol : str,
                       style : Dict[str, any] = {
                           "color" : "green",
                           "density" : True,
                       },
                       y_label : str = "Density",
                       y_log_scale : bool = True,
                       **kwargs):
        """Plot histogram of distribution and cleans old histogram

        Args:
            axes (plt.Axes): axes to plot onto
            symbol (str): stock to plot histogram
            style (Dict[str, any]): styles for plotting
            y_label (bool, optional): Label used for plot. Defaults to 'Density'.
            y_loc_scale (bool, optional): Whether to scale y axis by lograrithmus. Default to True.

        Raises:
            NotImplementedError: The function currently only works for Series
        """
        # restore points and label
        self.point_artist = None
        self.point_text_artist = None
        axes.clear()
        base_statistic.BaseStatistic.draw_histogram(self, axes, symbol, style, y_label, y_log_scale, **kwargs)