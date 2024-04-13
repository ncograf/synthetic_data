from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import quantile_statistic
import temporal_statistc


class SpikeStatistic(
    quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic
):
    def __init__(
        self,
        denomiator_scaling: Callable[[float], float],
        quantile: float,
        function_name: str = "func",
    ):
        """Initialize Spike statistic

        Args:
            denomiator_scaling (Callable[[float], float]): Monotonically increasing positive fuction [0,+\infty] -> \R_+
            name (str, optional): Name to be given to the statistic. Defaults to func.
        """
        temporal_statistc.TemporalStatistic.__init__(self)
        quantile_statistic.QuantileStatistic.__init__(self, quantile)

        self.denominator_scaling = np.vectorize(denomiator_scaling)

        self._name = (
            r"Spike statistic $\frac{|\Delta X_t - \Delta X_{t+1}|}{"
            + function_name
            + "(|X_t - X_{t+1}|)}$"
        )

    def set_statistics(self, data: pd.DataFrame | pd.Series):
        """Computes the wavelet statistic from the stock prices

        Args:
            data (Union[pd.DataFrame, pd.Series]): stock prices
        """
        self.check_data_validity(data)

        if isinstance(data, pd.Series):
            data = data.to_frame()

        self._symbols = data.columns.to_list()
        self._dates = data.index.to_list()
        self.statistic = self._spike_transform(data)

    def _spike_transform(self, series: npt.NDArray) -> npt.NDArray:
        """Spike transform

        The spike transform is defined as
        $sp(X_t) = \frac{|2X_{t} - X_{t-1} - X_{t_1}|}{scale(|X_{t+1} - X_{t-1}|)}$
        where scale is some scaling function defined in the constructor

        To get a full array, the result will be mean-padded

        Args:
            series (npt.NDArray): Data to be transformed

        Raises:
            ValueError: Only Series are supported at the moment

        Returns:
            npt.NDArray: Transformed values
        """

        if not isinstance(series, npt.NDArray) or series.ndim != 2:
            raise ValueError("Transform is only supported for numpy matrices")

        np_series = series
        nominator = 2 * np_series[1:-1] - np_series[2:] - np_series[:-2]

        denominator = np_series[2:] - np_series[:-2]
        denominator = self.denominator_scaling(
            np.abs(denominator)
        )  # TODO this abs is somehow weired

        spike_trans = nominator / (denominator + 1e-10)
        avg = np.mean(spike_trans, axis=0)
        spike_trans = np.pad(
            spike_trans, pad_width=(1, 0), constant_values=avg, mode="constant"
        )

        return spike_trans
