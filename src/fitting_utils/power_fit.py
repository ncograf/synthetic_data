from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


def fit_powerlaw(
    x: npt.NDArray, y: npt.NDArray, optimize: Literal["left", "right", "both", "none"]
) -> Tuple[float, float, float, float, float]:
    """Fit the powerlaw on a subset of the data

    The subset is chosen greedily to maximize the pearson coefficient

    The method fits
    :math:`y = a x^b \iff \log(y) = \log(a) + b \log(x)`
    using a linear regression in the log log space

    OPTIMIZE
        The parameter `optimize` determines the optimization method. To optimize,
        certain points (x,y) will be excluded. Available options are:
        - left : try all x = x[:-n] for n \in [50, 100, 150, ...] and choose optimal fit
        - right : try all x = x[n:] for n \in [50, 100, 150, ...] and choose optimal fit
        - both : greedy algorithm chooses at every iteration whether to cut from the right or left depending on the better fit
        - none : all points are included


    Args:
        x (npt.NDArray): x data
        y (npt.NDArray): y data
        optimize ('left' | 'right' | 'both' | 'none'): optimization style as descibed above

    Returns:
        Tuple[float, float, float, float, float]: a, b, pearson_coefficient, x, y
    """

    if optimize == "none":
        return x, y, *_fit_exp(x, y)

    elif optimize == "both":
        return _fit_powerlaw_both(x, y)

    elif optimize == "left":
        return _fit_powerlaw_left(x, y)

    elif optimize == "right":
        return _fit_powerlaw_right(x, y)

    else:
        raise RuntimeError(f"The otimization strategy {optimize} is not allowed.")


def fit_lin_log(x: npt.NDArray, y: npt.NDArray) -> Tuple[float, float, float]:
    """Fit a linlog regression

    The method fits
    :math:`y = a \exp(b x) \iff \log(y) = \log(a) + b x`
    using a linear regression in the lin space

    Args:
        x (npt.NDArray): x data
        y (npt.NDArray): y data

    Returns:
        Tuple[float]: a, b and pearson_coefficient
    """

    mask = y > 0
    x = x[mask]
    y = y[mask]

    log_y = np.log(y)

    fit = linregress(x, log_y)
    goodness = abs(fit.rvalue)
    slope = fit.slope
    intercept = fit.intercept

    return intercept, slope, goodness


def _fit_exp(x: npt.NDArray, y: npt.NDArray) -> Tuple[float, float, float]:
    """Fit the powerlaw on the data

    The method fits
    :math:`y = a  x^b \iff \log(y) = \log(a) + b \log(x)`
    using a linear regression in the log log space

    Args:
        x (npt.NDArray): x data
        y (npt.NDArray): y data

    Returns:
        Tuple[float]: a, b and pearson_coefficient
    """

    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    fit = linregress(log_x, log_y)
    goodness = abs(fit.rvalue)
    slope = fit.slope
    intercept = fit.intercept

    return intercept, slope, goodness


def _fit_powerlaw_left(
    x: npt.NDArray, y: npt.NDArray
) -> Tuple[float, float, float, float, float]:
    n_delta = 50
    n_bins = (x.size - 100) // n_delta

    fits = [(x, y, *_fit_exp(x, y))]
    for idx in range(1, n_bins):
        l_x, l_y = x[: -(n_delta * idx)], y[: -(n_delta * idx)]
        fits.append((l_x, l_y, *_fit_exp(l_x, l_y)))

    fit = max(fits, key=lambda x: x[-1])

    return fit


def _fit_powerlaw_right(
    x: npt.NDArray, y: npt.NDArray
) -> Tuple[float, float, float, float, float]:
    n_delta = 50
    n_bins = (x.size - 100) // n_delta

    fits = [(x, y, *_fit_exp(x, y))]
    for idx in range(1, n_bins):
        r_x, r_y = x[(n_delta * idx) :], y[(n_delta * idx) :]
        fits.append((r_x, r_y, *_fit_exp(r_x, r_y)))

    fit = max(fits, key=lambda x: x[-1])

    return fit


def _fit_powerlaw_both(
    x: npt.NDArray, y: npt.NDArray
) -> Tuple[float, float, float, float, float]:
    _, _, c_g = _fit_exp(x, y)
    n_g = 1

    while n_g > c_g:
        # make sure to have at least 500 points for fitting
        # note that this is heuristically optimied for 1000 bins!!!
        if 500 > x.size:
            return x, y, *_fit_exp(x, y)

        # determine change size
        n_delta = int(x.size * 0.06)

        # compute right and left option
        l_x, l_y = x[:-n_delta], y[:-n_delta]
        r_x, r_y = x[n_delta:], y[n_delta:]

        _, _, l_g = _fit_exp(l_x, l_y)
        _, _, r_g = _fit_exp(r_x, r_y)

        if l_g > r_g:
            x, y = l_x, l_y
            n_g = l_g
        else:
            x, y = r_x, r_y
            n_g = r_g

    return x, y, *_fit_exp(x, y)
