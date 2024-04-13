import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import shutil
import stylized_fact
from tueplots import bundles
import base_statistic


class StatisticInspector:
    """Class to clean data, visualize it and compute basic statistics
    for cleaning and augmentation purposes
    """

    def __init__(self, cache: str = "data/cache"):
        """Provide some tools to investigate the data

        Args:
            cache (str, optional): Directory to find cached files. Defaults to "data/cache".
        """

        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

        plt.rcParams["text.usetex"] = True

    def plot_average_sylized_fact(
        self,
        stylized_fact: stylized_fact.StylizedFact,
        rc_params: Optional[Dict[str, any]] = None,
        ax_params: Optional[Dict[str, any]] = None,
        style: Optional[Dict[str, any]] = None,
        copy: Optional[Path] = None,
    ):
        """Plots a histogram of the given samples statistic

        Args:
            statistic (base_statistic.BaseStatistic): statistic to be plotted
            symbol (str): stock symbol to plot histogram of
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            copy (Optional[Path]): path to copy the file to other than the cache directory
            density (bool, optional): Whether to show the density instead of the values. Defaults to True.
        """
        if rc_params is None:
            rc_params = bundles.icml2024()

        if ax_params is None:
            ax_params = {
                "yscale": "log",
                "xscale": "log",
                "xlim": (None, None),
                "ylim": (None, None),
                "xlabel": "None",
                "ylabel": "None",
            }

        fig_path = self.cache / f"{stylized_fact._figure_name}_average.png"

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            if style is None:
                style: Dict[str, any] = {
                    "alpha": 1,
                    "marker": "o",
                    "markersize": 1,
                    "linestyle": "None",
                    "color": "blue",
                    "color_neg": "red",
                    "color_pos": "blue",
                }
            stylized_fact.draw_stylized_fact_averaged(ax, style=style)
            self.draw_axis(ax=ax, ax_style=ax_params)
            fig.savefig(fig_path)

        if copy is not None:
            copy_path = copy / f"{stylized_fact._figure_name}_average.png"
            shutil.copy(fig_path, copy_path)

        plt.show()

    def plot_flipped_cfd(
        self,
        stat: base_statistic.BaseStatistic,
        rc_params: Optional[Dict[str, any]] = None,
        ax_params: Optional[Dict[str, any]] = None,
        copy: Optional[Path] = None,
    ):
        """Plots a histogram of the given samples statistic

        Args:
            statistic (base_statistic.BaseStatistic): statistic to be plotted
            symbol (str): stock symbol to plot histogram of
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            copy (Optional[Path]): path to copy the file to other than the cache directory
            density (bool, optional): Whether to show the density instead of the values. Defaults to True.
        """
        if rc_params is None:
            rc_params = bundles.icml2024()

        fig_path = self.cache / f"{stat._figure_name}_flipped_cfd.png"

        if ax_params is None:
            ax_params = {
                "yscale": "log",
                "xscale": "log",
                "xlim": (None, None),
                "ylim": (None, None),
                "xlabel": "None",
                "ylabel": "None",
            }

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            style: Dict[str, any] = {
                "alpha": 1,
                "marker": "o",
                "markersize": 1,
                "linestyle": "None",
            }
            stat.draw_flipted_cfd(ax, style=style)
            self.draw_axis(ax, ax_style=ax_params)
            fig.savefig(fig_path)

        if copy is not None:
            copy_path = copy / f"{stylized_fact._figure_name}_flipped_cfd.png"
            shutil.copy(fig_path, copy_path)

        plt.show()

    def draw_axis(
        self,
        ax: plt.Axes,
        ax_style: Dict[str, any] = {
            "yscale": "log",
            "xscale": "log",
            "xlim": (0, 1),
            "ylim": (0, 1),
            "xlabel": "None",
            "ylabel": "None",
        },
    ):
        ax.set(**ax_style)
