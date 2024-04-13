import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


class Plotter:
    def __init__(
        self,
        cache: Path | str,
        figure_name: str,
        figure_title: str,
        subplot_layout: Dict[str, any],
        figure_style: Dict[str, any] = {
            "figure.figsize": (16, 10),
            "font.size": 12,
            "figure.dpi": 96,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.1,
            "figure.constrained_layout.hspace": 0,
            "figure.constrained_layout.w_pad": 0.1,
            "figure.constrained_layout.wspace": 0,
        },
        export: Path | str | None = None,
    ):
        """Generates plotter class to simplify and normify plots for the project

        Args:
            cache (Path | str): cache directory to store the plots
            figure_name (str): figure name to store the plot in the cache
            figure_title (str): title to be plotted in the top
            subplot_layout (Dict[str, any]): size in terms of subplots
            figure_style (Dict[str, any], optional): Style, how the figure is supposed to look like. Defaults to { "figure.figsize" : (16, 10), "font.size" : 12, "figure.dpi" : 96, "figure.constrained_layout.use" : True, "figure.constrained_layout.h_pad" : 0.1, "figure.constrained_layout.hspace" : 0, "figure.constrained_layout.w_pad" : 0.1, "figure.constrained_layout.wspace" : 0, }.
            export (Path | str | None, optional): Export path if image must be exported to some other path. Defaults to None.
        """

        self._fig: plt.Figure
        self._axes: plt.Axes
        plt.rcParams.update(figure_style)
        self._fig, self._axes = plt.subplots(**subplot_layout, constrained_layout=True)
        self._fig.suptitle(figure_title)

        self._figure_title = figure_title
        self._figure_name = Path(figure_name).stem

        if export is not None:
            self._export_dir = Path(export)
        else:
            self._export_dir = None

        self._cache_dir = Path(cache)

    @property
    def export_path(self) -> Path | None:
        if self._export_dir is None:
            return None
        return self._export_dir / (self.figure_name + ".png")

    @property
    def cache_path(self) -> Path:
        return self._cache_dir / (self.figure_name + ".png")

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
        if self.export_path is not None:
            self._fig.savefig(self.export_path)
