from abc import abstractmethod, abstractproperty
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Set, List, Iterable
import pandas as pd
import numpy as np
import numpy.typing as npt
from tick import Tick
import base_outlier_set
from pathlib import Path
import json

class CachedOutlierSet(base_outlier_set.BaseOutlierSet):
    """Outlier Detector which only loads the csv file of outliers"""
    
    def __init__(self, path : Path | str | None = None):
        base_outlier_set.BaseOutlierSet.__init__(self)
        if not path is None and Path(path).exists():
            self.load_outliers(path)
        self._name = "Cached"
    
    @property
    def outlier(self):
        return self._outlier

    @outlier.setter
    def outlier(self, value : Iterable):
        value = set(value) # remove duplicates
        self._outlier = list(value) # make it sorted
        self._outlier.sort()

    @property
    def data(self) -> pd.DataFrame:
        """Data property containing the data on which the outliers are computed"""
        raise ValueError("No data available for chached set")
    
    def __len__(self):
        if self._outlier is None:
            return 0
        return len(self._outlier)

    def __getitem__(self, idx : int) -> Tick:
        return self._outlier[idx]

    def set_outlier(self, data : Set[Tick]):
        """store the outliers

        Args:
            data (pd.Series): Used to get the name of the data

        """
        if not isinstance(data, set):
            raise ValueError("Only sets are allowd as data")
        self.outlier = data
    
    def get_outlier(self) -> List[Tick]:
        """Computes a list of outlier points

        Args:
            **kwargs (any): any arguments used for the comuptation

        Returns:
            List[Tuple[pd.Timestamp, str]]: List of outliers points in the time series
        """
        self.check_outlier()
        return self.outlier

    def _to_dict_of_list(self):
        """Coverts outliers into dict of list"""

        _dict : Dict[str, List[Dict[str, pd.Timestamp | str | bool]]] = {}
        for tick in self.outlier:
            if not tick.symbol in _dict.keys():
                _dict[tick.symbol] = []
            
            _dict[tick.symbol].append(tick.get_dict())
        return _dict

    def _from_dict_of_list(self, dict : Dict[str, List[Dict[str, pd.Timestamp | str | bool]]]) -> Set[Tick]:
        """Converts Dictonary into Set of Ticks

        Args:
            dict (Dict[str, List[Dict[str, pd.Timestamp  |  str  |  bool]]]): Dictonary to be converted

        Returns:
            Set[Tick]: Set of ticks
        """

        _set = set()
        
        for symbol in dict:
            if not isinstance(symbol, str):
                raise RuntimeError("Symbols in dict must be string")

            for _elem in dict[symbol]:

                if not 'date' in  _elem.keys() or _elem['date'] is None:
                    raise RuntimeError("Dictonary must contain a valid date")
                date = pd.Timestamp(_elem['date']).floor('d')
                
                note = ''
                if 'note' in _elem.keys():
                    note = _elem['note']

                real = True
                if 'real' in _elem.keys():
                    real = _elem['real']
                
                _set.add(Tick(date, symbol, note, real))
            
        return _set

    def store_outliers(self, path : Path | str):
        """Stores set of outliers at the given path

        Args:
            path (Path | str): Path to store outliers at

        Raises:
            FileExistsError: Path to parent director must exists
        """

        path = Path(path)
        if not path.parent.exists():
            raise FileExistsError("The path to the parent must already exists.")

        if self.outlier is None or len(self.outlier) == 0:
            return 
        
        _dict = self._to_dict_of_list()

        with path.open("w") as target:
            json.dump(_dict, target)
    
    def load_outliers(self, path : Path | str):
        """Load outliers from backup file

        Args:
            path (Path | str): Path to look for the file

        Raises:
            FileExistsError: File at path must exist
            RuntimeError: File must have the right schema
        """

        path = Path(path)
        if not path.exists():
            raise FileExistsError(f"The path {str(path)} does not exist.")
        
        with path.open() as source:
            _dict = json.load(source)

        self.outlier = self._from_dict_of_list(_dict)
