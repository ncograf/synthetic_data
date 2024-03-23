import matplotlib.pyplot as plt
import numpy as np
import base_statistic
from typing import Dict

class StylizedFact(base_statistic.BaseStatistic):

    def __init__(self):
        self.color = 'blue'
        self.y_label = 'None'
        
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

        self.check_statistic()
        data = np.mean(self.statistic, axis=1)
        plt.plot(data, **style)