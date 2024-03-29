import pandas as pd
import data_inspector
import real_data_loader
from typing import List, Tuple
from pathlib import Path
import numpy as np

class TestConstructor:

    def test_constructor(self):
        #TODO Test reasonably
        test_df = pd.DataFrame([], columns=["A", "B", "C", "D"])
        test_df = pd.DataFrame([[1,2,3,5], [4,5,6,1], [7,8,9,10]], columns=["A", "B", "C", "D"])

if __name__ == "__main__":
    TestConstructor().test_constructor()