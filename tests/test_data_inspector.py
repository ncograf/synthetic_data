import pandas as pd
import data_inspector
import real_data_loader
from typing import List, Tuple
from pathlib import Path
import numpy as np

class TestConstructor:

    def test_constructor(self):
        test_df = pd.DataFrame([], columns=["A", "B", "C", "D"])

        try:
            inspector = data_inspector.DataInspector(data=test_df, cache="tests/testdata/cache")
            assert False, "Value Error expected"
        except ValueError as ex:
            assert "Data inputs not containing" in str(ex)

        test_df = pd.DataFrame([[1,2,3,5], [4,5,6,1], [7,8,9,10]], columns=["A", "B", "C", "D"])
        try:
            inspector = data_inspector.DataInspector(data=test_df, cache="tests/testdata/cache")
            assert False, "Value Error expected"
        except ValueError as ex:
            assert "Index must be time series" in str(ex)
