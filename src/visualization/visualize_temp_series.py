import matplotlib.pyplot as plt
import pandas as pd


def visualize_temp_data(
    temp_data: pd.DataFrame,
) -> plt.Figure:
    if isinstance(temp_data, pd.Series):
        temp_data = temp_data.to_frame()

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "figure.figsize": (16, 10),
        "figure.titlesize": 22,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "font.size": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 96,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    subplot_layout = {
        "ncols": 1,
        "nrows": 1,
        "sharex": "none",
        "sharey": "none",
    }

    plt.rcParams.update(figure_style)
    fig, axes = plt.subplots(**subplot_layout, constrained_layout=True)
    for col in temp_data.columns:
        axes.plot(temp_data.loc[:, col], label=col)
    axes.legend()

    return fig
