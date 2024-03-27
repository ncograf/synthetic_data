import matplotlib.pyplot as plt
import numpy as np
import base_statistic
from typing import Dict

class StylizedFact(base_statistic.BaseStatistic):

    def __init__(self):
        self.color = 'blue'
        self.y_label = 'None'
        self.x_ticks = None

    def draw_stylized_fact(
            self,
            ax : plt.Axes,
            style : Dict[str, any] = {
                'alpha' : 1,
                'marker' : 'o',
                'markersize' : 1,
                'linestyle' : 'None'
            }
            ):
        """Draws the averaged statistic over all symbols on the axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """
        
        if not 'color' in style.keys():
            style['color'] = self._plot_color

        self.check_statistic()
        data = np.mean(self.statistic, axis=1)
        ax.plot(data, **style)
        
    def draw_stylized_fact_averaged(
            self,
            ax : plt.Axes,
            style : Dict[str, any] = {
                'alpha' : 1,
                'marker' : 'o',
                'markersize' : 1,
                'linestyle' : 'None'
            }
            ):
        """Draws the averaged statistic over all symbols on the axes

        Args:
            ax (plt.Axes): Axis to draw onto
        """
        
        if not 'color' in style.keys():
            style['color'] = self._plot_color
        
        style.pop('color_neg')
        style.pop('color_pos')
        
        self.check_statistic()
        data = np.mean(self.statistic, axis=1)
        
        if self.x_ticks is None:
            ax.plot(data, **style)
        else:
            ax.plot(self.x_ticks, data, **style)
    