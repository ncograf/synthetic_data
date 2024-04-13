import numpy as np
import matplotlib.pyplot as plt
import click
from matplotlib.gridspec import GridSpec
import matplotlib
from getch import getch
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
import shutil
import base_statistic
import temporal_statistc
import base_outlier_set
import cached_outlier_set
import historic_events
from tueplots import bundles
import time
from tick import Tick


class DataInspector:
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

        matplotlib.use("qtagg")
        plt.rcParams["text.usetex"] = True

    def plot_time_series(
        self,
        statistics: List[List[temporal_statistc.TemporalStatistic]],
        symbols: List[str],
        rc_params: Optional[Dict[str, any]] = None,
        name: str = "time_series",
        copy: Optional[Path] = None,
    ):
        """Plots a histogram of the given samples statistic

        Args:
            statistics (List[base_statistic.BaseStatistic]): statistic to be plotted
            symbols (List[str]): stock symbol to plot histogram of
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            name (str, optionsl): Filename to store the plot
            copy (Optional[Path]): path to copy the file to other than the cache directory
        """
        if rc_params is None:
            rc_params = bundles.icml2024()

        fig_path = self.cache / f"{name}.png"

        n_subplots = len(statistics)

        with plt.rc_context(rc=rc_params):
            fig, axes = plt.subplots(n_subplots, 1)
            for idx, stats_sublist in enumerate(statistics):
                grow_only = False  # reset axis
                for statistic in stats_sublist:
                    for symbol in symbols:
                        style_plot = {
                            "linestyle": "-",
                            "linewidth": 1,
                        }
                        if n_subplots == 1:
                            statistic.draw_series(
                                axes,
                                symbol=symbol,
                                style_plot=style_plot,
                                grow_only=grow_only,
                            )
                        else:
                            statistic.draw_series(
                                axes[idx],
                                symbol=symbol,
                                style_plot=style_plot,
                                grow_only=grow_only,
                            )
                        grow_only = True
            plt.show()
            fig.savefig(fig_path)
        if copy is not None:
            copy_path = copy / f"{name}.png"
            shutil.copy(fig_path, copy_path)

    def plot_histogram(
        self,
        statistic: base_statistic.BaseStatistic,
        symbol: str | None,
        rc_params: Optional[Dict[str, any]] = None,
        copy: Optional[Path] = None,
        density: bool = True,
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

        fig_path = self.cache / f"{statistic._figure_name}_{symbol}_hist.png"

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            style: Dict[str, any] = {
                "color": statistic._plot_color,
                "density": True,
            }
            statistic.draw_histogram(ax, symbol=symbol, style=style, y_label="Denstiy")
            fig.savefig(fig_path)

        if copy is not None:
            copy_path = copy / f"{statistic._figure_name}_{symbol}_hist.png"
            shutil.copy(fig_path, copy_path)

        plt.show()

    def _iterate_outliers(
        self,
        statistics: List[base_statistic.BaseStatistic],
        time_statistics: List[Set[temporal_statistc.TemporalStatistic]],
        outlier_sets: List[base_outlier_set.BaseOutlierSet],
        historic_events: historic_events.HistoricEventSet,
        outlier_path: str | Path,
        fig: plt.Figure,
        ax_time: List[plt.Axes],
        ax_stats: List[plt.Axes],
        **kwargs,
    ) -> cached_outlier_set.CachedOutlierSet:
        if len(statistics) != len(ax_stats):
            raise ValueError(
                "For each statistic there must extactly one statistic axis"
            )

        if len(time_statistics) != len(ax_time):
            raise ValueError(
                "For each time statistic there must extactly one time axis"
            )

        outliers = [out_set.get_outlier() for out_set in outlier_sets]
        if len(outliers) > 0:
            all_outliers: Set[Tick] = set.union(
                *outliers
            )  # collect all outliers in one set to avoid duplicates
        else:
            all_outliers = set()

        outlier_cache = cached_outlier_set.CachedOutlierSet(outlier_path)
        outlier_cache.set_outlier(all_outliers)

        fig.canvas.flush_events()
        fig.canvas.draw()

        i: int = 0
        last_symbol: str = None
        while i < len(outlier_cache):
            point = outlier_cache[i]

            if last_symbol != point.symbol:
                for stat, ax in zip(statistics, ax_stats):
                    style: Dict[str, any] = {
                        "color": stat._plot_color,
                        "density": True,
                    }
                    stat.draw_histogram(axes=ax, style=style, symbol=point.symbol)

                for stat_set, ax in zip(time_statistics, ax_time):
                    for stat in stat_set:
                        stat.restore_time_plot(ax)

                outlier_cache.store_outliers(outlier_path)

            t = time.time()
            for obj, ax in zip(statistics, ax_stats):
                obj.draw_point(axes=ax, point=point)

            for stat_set, ax in zip(time_statistics, ax_time):
                t = time.time()
                grow_only = False
                for stat in stat_set:
                    stat.draw_and_emph_series(
                        ax=ax,
                        ticks=point,
                        window_size=200,
                        neighbour_points=True,
                        grow_only=grow_only,
                    )
                    grow_only = True

            historic_events.draw_events(ax=ax_time[0], date=point.date)

            fig.canvas.flush_events()

            print(
                (
                    f"Outlier {i+1}/{len(outlier_cache)} of {point.symbol}: {point.date}\n"
                    "Press y/n to confirm/reject glitch, h to go back, ctrl-c to cancel, "
                    "e to edit/add event, s to add current symbol to an event, i to open search about the day, any other key to show next."
                ),
                flush=True,
            )
            print(time.time() - t, " time all")
            key = getch()

            if key == "y":
                point.real = False
                print(f"MARKED GLITCH {point.symbol}, {point.date}", flush=True)
                i += 1  # show next element
            elif key == "e":
                historic_events.cmd_edit_event(date=point.date, symbol=point.symbol)
                historic_events.draw_events(ax_time[0], point.date)
                historic_events.store_events(path=historic_events._init_path)
                fig.canvas.flush_events()
            elif key == "i":
                historic_events.open_event_links(point.date, point.symbol)
            elif key == "s":
                historic_events.cmd_add_symbol(date=point.date, symbol=point.symbol)
                historic_events.store_events(path=historic_events._init_path)
            elif key == "f":
                # Fly to the index
                try:
                    start = click.prompt("Choose your starting point", type=int)
                    if start < 0 or start >= len(outlier_cache):
                        click.echo(
                            f"{start} is an invalid index, stay within {0} and {len(outlier_cache)}"
                        )
                    i = start
                except click.BadParameter:
                    click.echo("The input was invalid...")

            elif key == "n":
                if not point.real:
                    point.real = True
                    print(f"REMOVED MARK from {point.symbol}: {point.date}", flush=True)
                i += 1  # show next element
            elif key == "\x03":  # ctrl-c
                plt.close()
                exit()
            elif key == "h":  # go back
                if i > 0:
                    i -= 1  # show previous element
                    print(f"Go one back from {point.symbol}: {point.date}", flush=True)
            else:
                i += 1  # show next element

        return outlier_cache

    def check_outliers_manually(
        self,
        statistics: List[base_statistic.BaseStatistic],
        time_statistics: List[Set[temporal_statistc.TemporalStatistic]],
        outlier_detectors: Set[base_outlier_set.BaseOutlierSet],
        outlier_path: Union[str, Path],
        historic_events_path: Union[str, Path],
        **kwargs,
    ):
        """Visually inspect outliers manually and add notes to outliers if necessary

        Args:
            statistics (List[base_statistic.BaseStatistic]): Statistics to be shown as histogram
            time_statistics (List[Set[temporal_statistc.TemporalStatistic]]): Time Statistics to be show ans series, one axes for each group.
            outlier_detectors (Set[base_outlier_set.BaseOutlierSet]): Detectors determining the outliers
            outlier_path (Union[str, Path]): Path to store the outliers at.

        Raises:
            ValueError: _description_
        """

        outlier_path = Path(outlier_path)
        statistics = statistics
        time_statistics = time_statistics
        outlier_detectors = set(outlier_detectors)

        hist_events = historic_events.HistoricEventSet(historic_events_path)

        n_stats = len(statistics)
        n_time_stats_sets = len(time_statistics)

        # set ratio of time and stat
        n_col_time = 2
        n_col_stat = 1

        n_col = n_col_time
        if n_stats > 0:
            n_col += n_col_stat  # used to get ratio 2:1

        if n_time_stats_sets <= 0:
            raise ValueError(
                "There must be at least one time statistic for the visual inspection"
            )

        if n_stats > 0:
            n_rows = np.lcm(n_stats, n_time_stats_sets)
        else:
            n_rows = n_time_stats_sets

        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_col, figure=fig)

        ax_time = []
        for i in range(n_time_stats_sets):
            row_s = i * (n_rows // n_time_stats_sets)
            row_e = (i + 1) * (n_rows // n_time_stats_sets)
            ax = fig.add_subplot(gs[row_s:row_e, :n_col_time])
            ax.set_ylim(auto=True)
            ax.set_xlim(auto=True)
            ax_time.append(ax)

        ax_stats = []
        for i in range(n_stats):
            row_s = i * (n_rows // n_stats)
            row_e = (i + 1) * (n_rows // n_stats)
            ax_stats.append(
                fig.add_subplot(gs[row_s:row_e, n_col_time:])
            )  # adust ratio here

        plt.ion()
        plt.show()

        fig.axes.clear()

        # the join of the dictionaries here will be realtively smooth
        self._iterate_outliers(
            statistics=statistics,
            time_statistics=time_statistics,
            outlier_sets=outlier_detectors,
            outlier_path=outlier_path,
            historic_events=hist_events,
            fig=fig,
            ax_stats=ax_stats,
            ax_time=ax_time,
            **kwargs,
        )
        plt.close()
