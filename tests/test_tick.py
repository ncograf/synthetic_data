import pandas as pd
from tick import Tick


class TestTick:
    def test_properties(self):
        date = pd.Timestamp("2017-01-01 12:00:00")
        symbol = "Test"
        note = "Note test"
        real = False
        outlier = Tick(date, "Test", "Note test", real)

        assert outlier.date == date.floor("d")
        assert outlier.note == note
        assert outlier.symbol == symbol
        assert outlier.real == real

        outlier.real = True
        assert outlier.real

        outlier.note = "Second"
        assert outlier.note == "Second"

        outlier.symbol = "A"
        assert outlier.symbol == "A"

    def test_dict_out(self):
        date = pd.Timestamp("2017-01-01 12:00:00")
        note = "Note test"
        real = False
        outlier = Tick(date, "Test", "Note test", real)

        _dict = {
            "note": note,
            "date": date.floor("d").strftime("%Y-%m-%d"),
            "real": real,
        }

        dict_out = outlier.get_dict()
        for k in _dict.keys():
            assert k in dict_out.keys()
            assert dict_out[k] == _dict[k]
