import coarse_fine_volatility
import gain_loss_asymetry
import heavy_tails
import leverage_effect
import linear_unpredictability
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import volatility_clustering


def visualize_stylized_facts(log_returns: npt.ArrayLike) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        log_returns (array_like):  (n_timesteps x m_stocks) or (n_timesteps) return data.

    Returns:
        plt.Figure: matplotlib figure ready to plot / save
    """

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.figsize": (16, 10),
        "figure.titlesize": 22,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "font.size": 14,
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
        "ncols": 3,
        "nrows": 2,
        "sharex": "none",
        "sharey": "none",
    }

    plt.rcParams.update(figure_style)
    fig, axes = plt.subplots(**subplot_layout, constrained_layout=True)

    # prepare data
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(f"Log Returns has {log_returns.ndim} dimensions.")

    log_price = np.cumsum(log_returns, axis=0)

    linear_unpredictability.visualize_stat(axes[0, 0], log_returns, "", [])
    heavy_tails.visualize_stat(axes[0, 1], log_returns, "", [])
    volatility_clustering.visualize_stat(axes[0, 2], log_returns, "", [])
    leverage_effect.visualize_stat(axes[1, 0], log_returns, "", [])
    coarse_fine_volatility.visualize_stat(axes[1, 1], log_returns, "", [])
    gain_loss_asymetry.visualize_stat(axes[1, 2], log_price, "", [], fit=False)

    return fig
