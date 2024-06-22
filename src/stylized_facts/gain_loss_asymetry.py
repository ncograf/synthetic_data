import boosted_stats
import numpy as np
import numpy.typing as npt


def gain_loss_asymmetry(log_price: npt.ArrayLike, max_lag: int, theta: float):
    log_price = np.array(log_price)
    if log_price.ndim == 1:
        log_price = log_price.reshape((-1, 1))
    elif log_price.ndim > 2:
        raise RuntimeError(
            f"Log Price has {log_price.ndim} dimensions should have 1 or 2."
        )

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    if log_price.dtype.name == "float32":
        boosted = boosted_stats.gain_loss_asym_float(log_price, max_lag, theta, False)
    elif log_price.dtype.name == "float64":
        boosted = boosted_stats.gain_loss_asym_double(log_price, max_lag, theta, False)
    else:
        raise ValueError(f"Unsupported data type: {log_price.dtype.name}")

    boosted_gain = boosted[0] / boosted[0].sum(axis=0)
    boosted_loss = boosted[1] / boosted[1].sum(axis=0)
    return boosted_gain, boosted_loss


gain_loss_axis_setting = {
    "title": "gain loss asymetry",
    "ylabel": r"return time probability",
    "xlabel": "lag k",
    "xscale": "log",
    "yscale": "linear",
}

gain_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "red",
    "markersize": 2,
    "linestyle": "None",
    "label": r"gain $\theta > 0$",
}

loss_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "blue",
    "markersize": 1,
    "linestyle": "None",
    "label": r"loss $\theta < 0$",
}
