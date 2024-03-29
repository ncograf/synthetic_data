import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class Plotter:
    
    def __init__(
        self,
        cache : Path | str,
        figure_name : str,
        figure_title : str,
        subplot_layout : Tuple[int],
        figure_style : Dict[str, any] =  
        {
            "figure.figsize" : (16, 10),
            "font.size" : 11,
            "figure.dpi" : 96,
            "figure.constrained_layout.use" : True,
            "figure.constrained_layout.h_pad" : 0.1,
            "figure.constrained_layout.hspace" : 0,
            "figure.constrained_layout.w_pad" : 0.1,
            "figure.constrained_layout.wspace" : 0,
        },
        export : Path | str | None = None,
        ):
        """_summary_

        Args:
            cache (Path | str): Cache path to store image
            figure_name (str): figure name to store
            figure_title (str): figure title on image
            subplot_layout (Tuple[int]): layout (number and arrangement)
            figure_style (_type_, optional): _description_. Defaults to { "figure.figsize" : (16, 10), "font.size" : 24, "figure.dpi" : 96, "figure.constrained_layout.use" : True, "figure.constrained_layout.h_pad" : 0.1, "figure.constrained_layout.hspace" : 0, "figure.constrained_layout.w_pad" : 0.1, "figure.constrained_layout.wspace" : 0, }.
            export (Path | str | None, optional): Export path if necessary. Defaults to None.
        """
        
        self._fig : plt.Figure
        self._axes : plt.Axes
        with plt.rc_context(figure_style):
            self._fig, self._axes = plt.subplots(*subplot_layout, constrained_layout=True)
            self._fig.suptitle(figure_title)

        self._figure_title = figure_title
        self._figure_name = Path(figure_name).stem

        if not export is None:
            self._export_dir = Path(export)
        else:
            self._export_dir = None

        self._cache_dir = Path(cache) 

    @property
    def export_path(self) -> Path | None:
        if self._export_dir is None:
            return None
        return self._export_dir / (self.figure_name + '.png')

    @property
    def cache_path(self) -> Path:
        return self._cache_dir / (self.figure_name + '.png')
    
    @property
    def figure_name(self) -> str:
        return self._figure_name

    @property
    def figure(self) -> plt.Figure:
        return self._fig

    @property
    def axes(self) -> plt.Axes:
        return self._axes
        
    def save(self):
        """Saves the figure to the cache and export path if given"""
        
        self._fig.savefig(self.cache_path)
        if not self.export_path is None:
            self._fig.savefig(self.export_path)

    
    