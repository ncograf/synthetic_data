from typing import Dict, List

import base_statistic
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class StylizedFact(base_statistic.BaseStatistic):
    def __init__(self):
        """Initialize stylized"""
        self._x_ticks = None
        self._ax_style = {
            "title": "No Title",
            "y_label": "None",
            "x_label": "None",
        }
        self._styles: List[Dict[str, any]] = [{}]

    @property
    def x_ticks(self) -> npt.NDArray | None:
        """Data for the x axis"""
        return self._x_ticks

    @property
    def ax_style(self) -> Dict[str, any]:
        """Axis style properties, such as scale and labels"""
        return self._ax_style

    @property
    def styles(self) -> List[Dict[str, any]]:
        """Get style

        Returns:
            List[Dict[str, any]]: List of styles
        """
        return self._styles

    @styles.setter
    def styles(self, value: List[Dict[str, any]]):
        """Set style"""
        self._styles = value

    def draw_stylized_fact(self, ax: plt.Axes, **kwargs):
        """Draws the averaged statistic over all symbols on the axis

        Args:
            ax (plt.Axes): Axis to plot on
        """
        self.check_statistic()
        data = np.mean(self.statistic, axis=1)

        ax.set(**self.ax_style)
        if self.x_ticks is None:
            ax.plot(data, **self.styles[0])
        else:
            ax.plot(self.x_ticks, data, **self.styles[0])
