import boosted_stats
import numpy as np
import numpy.typing as npt


def leverage_effect(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Leverage effect

    Computes the correlation between current returns and future volatility (squared returns)

    Args:
        log_returns (npt.ArrayLike): log returns
        max_lag (int): maximal lag

    Raises:
        RuntimeError: Wrong dimension

    Returns:
        ndarray: (max_lag x stocks) leverage effects for different lags
    """

    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    if log_returns.dtype.name == "float32":
        stat = boosted_stats.leverage_effect_float(log_returns, max_lag, False)
    elif log_returns.dtype.name == "float64":
        stat = boosted_stats.leverage_effect_double(log_returns, max_lag, False)

    return stat


lev_eff_axes_setting = {
    "title": "leverage effect",
    "ylabel": r"$L(k)$",
    "xlabel": "lag k",
    "xscale": "linear",
    "yscale": "linear",
}
lev_eff_plot_setting = {
    "alpha": 1,
    "marker": "None",
    "color": "blue",
    "markersize": 0,
    "linestyle": "-",
}
