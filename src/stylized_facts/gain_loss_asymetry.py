import boosted_stats
import numpy as np
import pandas as pd


def gain_loss_asymmetry(price: pd.DataFrame, max_lag: int, theta: float):
    if isinstance(price, pd.Series):
        price = price.to_frame()

    price = np.array(price)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    log_price = np.log(price)
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
