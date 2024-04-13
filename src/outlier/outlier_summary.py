import pandas as pd
import base_outlier_set
from typing import List, Dict, Set
from tick import Tick


class OutlierSummary:
    def __init__(
        self, data: pd.DataFrame, detectors: List[base_outlier_set.BaseOutlierSet]
    ):
        self._data = data
        self._detectors = detectors

        all_outlier_dict: Dict[str, Set[Tick]] = {}

        # very ineficienent computation, but works
        for detector in detectors:
            detector.set_outlier(data)
            all_outlier_dict[detector._name] = detector.get_outlier()

        self.outlier_dict = all_outlier_dict

    def print_outlier_distribution(self):
        """Print sum of outliers, data inspected"""

        all_outlier_set = set()
        for det in self.outlier_dict:
            outlier_set = self.outlier_dict[det]
            all_outlier_set = set.union(all_outlier_set, outlier_set)
            self._print_one_sample(outlier_set, det)

        print()  # new line in between
        self._print_one_sample(all_outlier_set, "all detectors")

    def _print_one_sample(self, sample: Set[Tick], name: str):
        """Prints some simples numbers of the fetched outliers"""
        dates, symbols = Tick.unzip(sample)
        symbols = pd.Series(symbols)
        n_outlier = len(dates)
        symbol_groups = symbols.groupby(by=symbols).count()
        print(f"Outliers in {name}")
        print(f"Total: {n_outlier:>20}")
        print(f"Average per symbol: {symbol_groups.mean():>20}")
