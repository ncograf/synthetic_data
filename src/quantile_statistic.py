import base_statistic
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Optional, List
import pandas as pd
import numpy as np
import numpy.typing as npt


class QuantileStatistic(base_statistic.BaseStatistic):
        
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
        
    def set_statistics(self, data: Union[pd.DataFrame, pd.Series]):
        """Sets the statistic to given data

        Args:
            data (Union[pd.DataFrame, pd.Series]): statistic samples
        """

        self._check_data_validity(data)
        self.statistic = data

    def get_additional_stats(self, point: Tuple[pd.Timestamp | str]) -> Dict[str, float]:
        return {}
        
    def draw_distribution(self,
                          axes : plt.Axes,
                          **kwargs):
        
        self._check_statistics()
        
        if not isinstance(self.statistic, pd.Series):
            raise NotImplementedError("Drawing currently only works with series")

        # plot as histogram
        n_bins = np.minimum(self.statistic.shape[0], 80)
        color = "green"
        axes.hist(x=self.statistic, bins=n_bins, color=color, density=True)
        axes.set_label("Distribution")
        axes.set_xlabel(self._name)
        axes.set_ylabel("Density")
        
    def draw_point(self,
                   axes : plt.Axes,
                   index: Tuple[pd.Timestamp, str],
                   point_array : List[Tuple[pd.Timestamp, str]]
                   ):
                   
        self._check_statistics()

        if not isinstance(self.statistic, pd.Series):
            raise NotImplementedError("Drawing currently only works with series")
        
        emph_color = "red"
        normal_color = "green"
        emph_alpha = 0.5

        # get bin containing the index
        if self.point in axes.lines:
            x = self.statistic.at[index[0]]
            self.point.set_xdata(x=x)
        else:
            x = self.statistic.at[index[0]]
            self.point = axes.axvline(x = x, color=emph_color, alpha=emph_alpha, label=f"{index[1]} at {index[0].strftime('%Y-%m-%d')}")
            ylim = axes.get_ylim()
            self.text = axes.text(x,0.5 * (ylim[1] + ylim[0]), s="")

        if index in point_array:
            self.text.set_text("Outlier")
            self.text.set_color(emph_color)
            self.point.set_color(emph_color)
            self.text.set_x(x)
        else:
            self.text.set_text("No Outlier")
            self.text.set_color(normal_color)
            self.point.set_color(normal_color)
            self.text.set_x(x)

        handles, labels = axes.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        axes.legend(unique_labels.values(), unique_labels.keys())