from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import quantile_statistic
import scipy.linalg as linalg
import temporal_statistc


class Wavelet:
    def __init__(self, pattern: npt.ArrayLike, center: Optional[int] = None):
        """Wavelet representing base function for wavelet transform

        In other words the pattern indicates the <kernel> for a convolution
        on a finite domain, where the padding in specified by center. And padding
        is done by copying the outermost value

        Args:
            pattern (npt.ArrayLike): Wavelet
            center (Optional[int], optional): Center in the pattern (middle if None). Defaults to None.

        Raises:
            ValueError: Only series are supported and center must be in pattern
        """

        self.pattern = np.array(pattern, dtype=np.float32)

        if not (self.pattern.ndim == 1):
            raise ValueError("Only 1D Wavelets are supported")

        if center is None:
            self.center = self.pattern.shape[0] // 2
        else:
            self.center = center

        if self.center < 0 or self.center >= self.pattern.shape[0]:
            raise ValueError("Center must be in patterns support")


class WaveletStatistic(
    quantile_statistic.QuantileStatistic, temporal_statistc.TemporalStatistic
):
    def __init__(
        self,
        wavelet: Wavelet,
        normalize: bool,
        quantile: float,
        name: Optional[str] = None,
    ):
        quantile_statistic.QuantileStatistic.__init__(self, quantile)
        temporal_statistc.TemporalStatistic.__init__(self)

        self.normalize = normalize
        self.wavelet = wavelet

        if name is None:
            self._name = "Wavelet with support " + str(self.wavelet.pattern.shape[0])
        else:
            self._name = name

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
        self.statistic = self._wavelet_transform(data.to_numpy())

    def _wavelet_transform(self, series: npt.NDArray) -> npt.NDArray:
        """Wavlet transform applied on the given series

        Runtime O(n log(n)) by exploiting fft and a compact representation
        of the toepliz matrix.

        Args:
            series (npt.NDArray): Data to be transformed

        Raises:
            ValueError: Only Series are supported at the moment

        Returns:
            npt.NDArray: Transformed values
        """

        if not isinstance(series, npt.NDArray) and series.ndim == 2:
            raise ValueError("Transform only works for numpy matrices")

        n_series = series.shape[0]

        n_wavelet = self.wavelet.pattern.shape[0]
        n_wave_right = n_wavelet - self.wavelet.center
        n_wave_left = self.wavelet.center + 1

        # first row of the toepliz matirx (must include the center)
        toepliz_c = np.zeros(n_series)
        toepliz_c[:n_wave_right] = self.wavelet.pattern[self.wavelet.center :]

        # last row of the toepliz matrix (must include the center)
        toepliz_r = np.zeros(n_series)
        toepliz_r[:n_wave_left] = np.flip(self.wavelet.pattern[:n_wave_left])

        # compute transformation with toepliz transform
        trans_series = linalg.matmul_toeplitz((toepliz_c, toepliz_r), series)

        if self.normalize:
            raise NotImplementedError(
                "This function is depricated and needs serious revision"
            )
            # padding by COPY the EDGE VALUE to get a variance for the whole series
            # padded_array = np.pad(series, (n_wave_left, n_wave_right), mode='edge')
            # rolling_std = pd.Series(padded_array).rolling(window=n_wavelet).std() # comupute rolling stds
            # rolling_std = rolling_std.to_numpy()[n_wavelet + 1:]
            # rolling_mean = pd.Series(padded_array).rolling(window=n_wavelet).mean() # comupute rolling stds
            # rolling_mean = rolling_mean.to_numpy()[n_wavelet + 1:]

            # eps = 1e-2
            # trans_series = trans_series / series.abs() # normalize
            # trans_series[:n_wave_left + 1] = 0
            # trans_series[-n_wave_right - 1:] = 0 # add zero padding to avoid problems at the borders

        return trans_series
