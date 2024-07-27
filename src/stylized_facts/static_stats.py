from typing import Dict

import numpy as np
import numpy.typing as npt


def static_stats(log_returns: npt.ArrayLike) -> Dict[str, float]:
    """Compute a series of static statistics for the data

    Args:
        log_returns (npt.ArrayLike): data points

    Returns:
        Dict[str, float]: a dict of named statistics
    """

    data = np.asarray(log_returns).flatten()
    data = data[~np.isnan(data)]
    mean = np.nanmean(data)
    std = np.nanstd(data)
    stats = {
        "mean": mean,
        "std": std,
        "variance": np.nanvar(data),
        "skewness": np.nanmean((data - mean) ** 3) / (std**3),
        "kurtosis": np.nanmean((data - mean) ** 4) / (std**4),
        "min": np.nanmin(data),
        "max": np.nanmax(data),
        "median": np.nanmedian(data),
    }

    return stats
