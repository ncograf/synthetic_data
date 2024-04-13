from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pandas as pd


class Tick:
    def __init__(
        self, date: pd.Timestamp, symbol: str, note: str = "", real: bool = True
    ):
        self._date = pd.Timestamp(date)
        self._symbol = symbol
        self._note = note
        self._real = real

    def __eq__(self, other: Tick):
        """neglect real and note properties in comparison"""
        return self.date == other.date and self.symbol == other.symbol

    def __lt__(self, other):
        return (self.symbol.lower(), self.date) < (other.symbol.lower(), other.date)

    def __le__(self, other):
        return (self.symbol.lower(), self.date) <= (other.symbol.lower(), other.date)

    def __gt__(self, other):
        return (self.symbol.lower(), self.date) > (other.symbol.lower(), other.date)

    def __ge__(self, other):
        return (self.symbol.lower(), self.date) >= (other.symbol.lower(), other.date)

    def __hash__(self) -> int:
        return tuple.__hash__((self.date, self.symbol))

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        self._symbol = value

    @property
    def date(self) -> pd.Timestamp:
        return self._date.floor("d")

    @date.setter
    def date(self, value: pd.Timestamp):
        self._date = value

    @property
    def note(self) -> str:
        return self._note

    @note.setter
    def note(self, value: str):
        self._note = value

    @property
    def real(self) -> bool:
        """Describes whether the datapoint corresponds to a real datapoint as opposed to some glitch"""
        return self._real

    @real.setter
    def real(self, value: bool):
        self._real = value

    def get_dict(self) -> Dict[str, pd.Timestamp | str | bool]:
        """Returns the tick as dictonary without the symbol

        Returns:
            Dict[str, pd.Timestamp | str | bool]: dictonary without symbol
        """
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "note": self.note,
            "real": self.real,
        }

    @staticmethod
    def zip(dates: List[pd.Timestamp], symbols: List[str]) -> Set[Tick]:
        if len(dates) != len(symbols):
            raise ValueError("The indices and columns must have the same size")

        _set = set()
        for date, symbol in zip(dates, symbols):
            _set.add(Tick(date=date, symbol=symbol))

        return _set

    @staticmethod
    def unzip(
        ticks: Set[Tick], filter_unreal: bool = True
    ) -> Tuple[List[pd.Timestamp], List[str]]:
        """Returns dates and symbols as tuples, given a set of Ticks

        Args:
            ticks (Set[Tick]): set to extract dates and symbols
            filter_unreal (bool, optional): whether or not to filter. Defaults to True.

        Returns:
            Tuple[List[pd.Timestamp], List[str]]: Tuple with a list of dates and a list of symbols
        """

        dates = []
        symbols = []

        for tick in ticks:
            if not filter_unreal or tick.real:
                dates.append(tick.date)
                symbols.append(tick.symbol)

        return (dates, symbols)
