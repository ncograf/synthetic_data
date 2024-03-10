import base_statistic
import outlier_detector
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Optional, List, Set
import pandas as pd
import numpy as np
import numpy.typing as npt


class QuantileStatistic(base_statistic.BaseStatistic, outlier_detector.OutlierDetector):
    """Statistic with additional qualitative and quantitative measures
    to find outliers.
    """
    
    def __init__(self):
        super(QuantileStatistic, self).__init__()
        self._outlier = None
    
    @property
    def data(self):
        return self.statistic

    def _get_index(self, mask : npt.ArrayLike) -> Set[Tuple[any, any]]:
        """ Compute a list of indices and columns from mask

        Args:
            mask (npt.ArrayLike): mask of which to get the indices

        Returns:
            List[Tuple[any, any]]: List with indices where the mask is true
        """
        
        self._check_statistics()

        if isinstance(self.statistic, pd.DataFrame):
            np_indices = np.where(mask)
            indices = self.statistic.index[np_indices[0]]
            columns = self.statistic.columns[np_indices[1]]
        elif isinstance(self.statistic, pd.Series):
            np_indices = np.where(mask) # transfrom mask to list
            indices = self.statistic.index[np_indices]
            columns = [self.statistic.name] * len(indices)
        else:
            raise ValueError("Unsuported type in statistics. Required to be DataFrame or Series")

        return set(zip(indices, columns))
        
    def _get_mask(self, quantile : float, **kwargs) -> npt.NDArray:
        """Compute outliers based on the value

        Args:
            quantile (float): number of outliers on each side

        Raises:
            ValueError: Data must have datetime indices, quantile must be in (0,1) and
                columns, if existent, must be of type string

        Returns:
            Union[List[Tuple[str, np.datetime64]], List[str]]: List containing the index (and column depending on the data)of outliers
        """
        
        self._check_statistics()
        
        # check argument
        if quantile <= 0 or quantile >= 1:
            raise ValueError("The quantile must be within 0 and 1")

        # compute thesholds
        np_data = self.statistic.to_numpy()
        bot_thresh = np.nanpercentile(np_data.flatten(), quantile * 100)
        top_thresh = np.nanpercentile(np_data.flatten(), 100 * (1 - quantile))
        
        # get indices of outliers
        mask = (self.statistic < bot_thresh) | (self.statistic > top_thresh)

        return mask
    
    def get_outliers(self, **kwargs) -> Set[Tuple[pd.Timestamp, str]]:
        """Computes a list of outlier points in the dataframe / series

        If dataframe a list of tuples is returned, otherwise a list of indices

        Args:
            **kwargs (any): arguments which will be passed down to the mask

        Returns:
            List[Tuple[pd.Timestamp, str]]: List with indices where the mask is true
        """
        if self._outlier is None:
            mask = self._get_mask(**kwargs)
            self._outlier = self._get_index(mask=mask)
        return self._outlier
    
    def is_outlier(self, point: Tuple[pd.Timestamp | str]) -> bool:
        """Check whether point is in the outlier array

        Args:
            point (Tuple[pd.Timestamp  |  str]): point to check

        Raises:
            IndexError: Outlier not yet computed

        Returns:
            bool: Point is in outliers of this statistic
        """
        if self._outlier is None:
            raise IndexError("Outlier must be computed wiht get_outlier before checking for outlier")
        return point in self._outlier
        
    def set_statistics(self, data: Union[pd.DataFrame, pd.Series]):
        """Sets the statistic to given data

        Args:
            data (Union[pd.DataFrame, pd.Series]): statistic samples
        """

        self._check_data_validity(data)
        self.statistic = data
        
    def draw_point(self,
                   axes : plt.Axes,
                   point: Tuple[pd.Timestamp, str],
                   outlier_color : str = "red",
                   normal_color : str = "green",
                   outlier_alpha : float = 1,
                   normal_alpha : float = 1,
                   ):
        """Draws a single datapoint as a vertical line

        The plot can e.g. be the histogram plot in which it makes sense

        Args:
            axes (plt.Axes): Plot to draw the line into
            point (Tuple[pd.Timestamp, str]): datapoint (date, symbol) to be plotted
            outlier_array (List[Tuple[pd.Timestamp, str]]): list of points to be considered outliers
            outlier_color (str): Color to emphazise point if it is in the outlier array
            normal_color (str): Color to draw point in if it is not in outlier array
            outlier_alpha (float): Alpha in case point is in outlier array
            normal_alpha (flaot): Alpha in case point is not in outlier array

        Raises:
            NotImplementedError: Only (time)Series data is supported
        """
                   
        self._check_statistics()

        if not isinstance(self.statistic, pd.Series):
            raise NotImplementedError("Drawing currently only works with series")
        
        # get bin containing the index
        if self.point in axes.lines:
            x = self.statistic.at[point[0]]
            self.point.set_xdata(x=x)
        else:
            x = self.statistic.at[point[0]]
            self.point = axes.axvline(x = x, color=outlier_color, alpha=outlier_alpha, label=f"{point[1]} at {point[0].strftime('%Y-%m-%d')}")
            ylim = axes.get_ylim()
            self.text = axes.text(x,0.5 * (ylim[1] + ylim[0]), s="")

        if self.is_outlier(point=point):
            self.text.set_text("Outlier")
            self.text.set_color(outlier_color)
            self.text.set_alpha(outlier_alpha)
            self.point.set_color(outlier_color)
            self.point.set_alpha(outlier_alpha)
            self.text.set_x(x)
        else:
            self.text.set_text("No Outlier")
            self.text.set_color(normal_color)
            self.text.set_alpha(normal_alpha)
            self.point.set_color(normal_color)
            self.point.set_alpha(normal_alpha)
            self.text.set_x(x)

        handles, labels = axes.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        axes.legend(unique_labels.values(), unique_labels.keys())