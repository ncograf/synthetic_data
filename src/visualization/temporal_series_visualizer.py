from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotter
import temporal_statistc


def visualize_time_series(
    time_series: pd.DataFrame,
    styles: List[Dict[str, str]],
    figure_name: str = "time-series",
) -> plotter:
    temp_stat = temporal_statistc.TemporalStatistic()
    temp_stat.set_statistics(time_series)

    plot = plotter.Plotter(
        cache=Path(__file__).parent.parent / "data/cache",
        figure_style={
            "figure.figsize": (16, 10),
            "figure.titlesize": 24,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "font.size": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.dpi": 96,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.1,
            "figure.constrained_layout.hspace": 0,
            "figure.constrained_layout.w_pad": 0.1,
            "figure.constrained_layout.wspace": 0,
        },
        figure_name=figure_name.replace("-", "_"),
        figure_title=figure_name,
        subplot_layout={
            "ncols": 1,
            "nrows": 1,
            "sharex": "all",
            "sharey": "all",
        },
    )
    n_styles = len(styles)
    for i, symbol in enumerate(temp_stat.symbols):
        temp_stat.draw_series(plot.axes, symbol=symbol, style_plot=styles[i % n_styles])

    return plot
