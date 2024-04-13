from typing import Dict, List

import base_outlier_set
import base_statistic
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tick import Tick


class IsolationForestStatisticSet(base_outlier_set.BaseOutlierSet):
    """Outlier Detector class"""

    def __init__(self, quantile: float, statistic: base_statistic.BaseStatistic):
        base_outlier_set.BaseOutlierSet.__init__(self)

        # check argument
        if quantile <= 0 or quantile >= 1:
            raise ValueError("The quantile must be within 0 and 1")

        self._statistic = statistic
        self.contamination = quantile
        self._forests: Dict[str, IsolationForest] = {}
        self._name = "Isolation Forest"

    def set_outlier(self, data: pd.DataFrame):
        """Compute the outliers based on the statistics which
        has to be set in the constructor and comptuted before calling
        this function

        Args:
            data (pd.DataFrame): ignored

        """
        self._statistic.check_statistic()
        outlier_mask = np.zeros(self._statistic.statistic.shape[:2], dtype=bool)

        for col in self._statistic.symbols:
            col_idx = self._statistic.get_symbol_index(col)
            data = self._statistic.statistic[:, col_idx]
            all_nan = np.isnan(data).all(data.shape[1:])
            data = data[~all_nan]
            assert data.ndim == 1
            index = np.arange(data.shape[0])
            data = np.stack([index, data], axis=1)
            self._forests[col] = IsolationForest(contamination=self.contamination)
            outlier_mask[:, col_idx] = self._forests[col].fit_predict(data)
            dates = np.array(self._statistic.dates)[outlier_mask].tolist()
            columns = [col] * len(dates)

        self._outlier = Tick.zip(dates=dates, symbols=columns)

    def get_outliers(self, **kwargs) -> List[Tick]:
        """Computes a list of outlier points

        Args:
            **kwargs (any): any arguments used for the comuptation

        Returns:
            List[Tuple[pd.Timestamp, str]]: List of outliers points in the time series
        """
        self.check_outlier()

        return self._outlier
