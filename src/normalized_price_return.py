import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Dict, Optional, Literal
from icecream import ic
import matplotlib.pyplot as plt
import temporal_statistc
import scipy.linalg as linalg
import stylized_fact

class NormalizedPriceReturn(stylized_fact.StylizedFact):
    
    def __init__(
            self,
            underlaying : temporal_statistc.TemporalStatistic,
    ):
        
        stylized_fact.StylizedFact.__init__(self)

        self._name = r"$P(r) = P(r_t > r)$"
        self._sample_name = underlaying.name
        self._figure_name = "auto_correlation"
        self._underlaying = underlaying
        self._plot_color = 'blue'
        self.y_label = 'lag k'

    def normalized_returns(self) -> npt.NDArray:
        """ Compute the empirical 1 - F(X) for the statistic X

        Args:
            symbol (str | None, optional): Symbol to compute it for, computed over all stocks if None. Defaults to None.

        Returns:
            npt.NDArray: Array containing the list of values of X in a first column and (1 - F(X)) in a second column
        """
        self._underlaying.check_statistic()
        nan_mask = np.isnan(self._underlaying.statistic)
        ge_0 = self._underlaying.statistic > 0
        data_pos = self._underlaying.statistic[ge_0 & (~nan_mask)].flatten()
        data_neg = np.abs(self._underlaying.statistic[(~ge_0) & (~nan_mask)].flatten())

        k = 5000
        
        m_pos = data_pos.shape[0]
        if m_pos > 0:
            data_pos = np.sort(data_pos)
            std_pos = np.var(data_pos)
            mu_pos = np.mean(data_pos)
            norm_data_pos = (data_pos - mu_pos) / std_pos
            disp_data_pos = norm_data_pos[:: m_pos // k]
            n_pos = disp_data_pos.shape[0]
            P_r_pos =  1 - np.arange(1,n_pos+1) / n_pos
        else:
            disp_data_pos = np.array([])
            P_r_pos =  1 - np.array([])

        m_neg = data_neg.shape[0]
        if m_neg > 0:
            data_neg = np.sort(data_neg)
            std_neg = np.var(data_neg)
            mu_neg = np.mean(data_neg)
            norm_data_neg = (data_neg - mu_neg) / std_neg
            disp_data_neg = norm_data_neg[:: m_neg // k]
            n_neg = disp_data_neg.shape[0]
            P_r_neg =  1 - np.arange(1,n_neg+1) / n_neg
        else:
            disp_data_neg = np.array([])
            P_r_neg =  1 - np.array([])

        n = np.maximum(disp_data_neg.shape[0], disp_data_pos.shape[0])
        out = np.zeros((n,4))
        out[:] = np.nan
        out[:P_r_neg.shape[0],0] = P_r_neg
        out[:disp_data_neg.shape[0],1] = disp_data_neg
        out[:P_r_pos.shape[0],2] = P_r_pos
        out[:disp_data_pos.shape[0],3] = disp_data_pos

        return  out

    
    def set_statistics(self, data: pd.DataFrame | pd.Series | None = None):
        
        norm_ret = self.normalized_returns()
        self._statistic = norm_ret

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

        self.check_statistic()
        
        if not 'color' in style.keys():
            style['color'] = self._plot_color
        
        color = style['color']
        color_neg = style['color_neg'] if 'color_neg' in style.keys() else color
        color_pos = style['color_pos'] if 'color_pos' in style.keys() else color
        style.pop('color_pos')
        style.pop('color_neg')

        style['color'] = color_neg
        ax.plot(self.statistic[:,3], self.statistic[:,2], **style, label=r'$r_t < 0$')
        style['color'] = color_pos
        ax.plot(self.statistic[:,1], self.statistic[:,0], **style, label=r'$r_t > 0$')
        ax.legend()