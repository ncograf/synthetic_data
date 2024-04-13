import real_data_loader as data
from pathlib import Path
import pandas as pd
import shutil

class TestLocalData:
    """Test Function to get local data"""

    def test_local_data_basic(self):
        
        data_loader = data.RealDataLoader()
        data_dict, error_dict = data_loader._get_local_data(["ABBV", "ABNB", "ABT", "ACGL"], "tests/testdata/local", required_columns=["Adj Close", "Close"])
        
        assert len(data_dict) == 4
        assert error_dict == {}
        
        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"
        assert len(data_dict["ABBV"]) == 49
        assert set(data_dict["ABBV"].columns) == set(['Open','High','Low','Close','Adj Close','Volume'])

    def test_local_data_error(self):
        
        data_loader = data.RealDataLoader()
        symbols = ["ABBV", "ABNB", "ABT", "ACGL", "TEST", "TEST_EMPTY", "TEST_NO_DATA", "TEST_NEXIST"]
        data_dict, error_dict = data_loader._get_local_data(symbols, "tests/testdata/local", required_columns=["Adj Close"])
        
        assert len(data_dict) == 4
        assert len(error_dict) == 4
        
        assert set(error_dict.keys()) == set(["TEST", "TEST_EMPTY", "TEST_NO_DATA", "TEST_NEXIST"])

        message = ("The file for the symbol TEST only contains "
                    "the columns {'Date'} of the required columns")
        assert message in error_dict["TEST"]

        message = "Problem reading csv with pandas for symbol TEST_EMPTY"
        assert message in error_dict["TEST_EMPTY"]

        message = ("The file for the symbol TEST_NO_DATA contains not data.")
        assert message in error_dict["TEST_NO_DATA"]

        message = "For the symbol TEST_NEXIST no csv file was found!"
        assert message in error_dict["TEST_NEXIST"]
        
class TestYahooData:
    """Test Function to get yahoo data"""

    def test_yahoo_basic(self):

        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_dict = data_loader._get_yahoo_data(["ABBV", "ACGL"])

        assert len(data_dict) == 2
        assert error_dict == {}
        
        assert data_dict["ABBV"].index.name == "Date"
        assert set(data_dict["ABBV"].columns) == set(['Open','High','Low','Close','Adj Close','Volume'])
        
    def test_yahoo_error(self):

        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_dict = data_loader._get_yahoo_data(["TEST_NEXIST", "ACGL"])

        assert len(data_dict) == 1
        assert len(error_dict) == 1

        message =  "Problem downloading symbol TEST_NEXIST"
        assert message in error_dict["TEST_NEXIST"]

class TestGetAllData:

    def test_all_only_local(self):

        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_local, error_yahoo = data_loader._get_all_data(["ABBV", "ACGL"], data_path="tests/testdata/local", store_downloads=False)
        
        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"

        assert len(data_dict) == 2
        assert len(error_local) == 0
        assert len(error_yahoo) == 0

    def test_all_only_yahoo(self):

        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_local, error_yahoo = data_loader._get_all_data(["AFL", "A"], data_path="tests/testdata/local", store_downloads=False)
        
        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"

        assert len(data_dict) == 2
        assert len(error_local) == 2
        assert len(error_yahoo) == 0

    def test_all_mixed(self):

        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_local, error_yahoo = data_loader._get_all_data(["ABBV", "AIZ"], data_path="tests/testdata/local")
        Path("tests/testdata/local/AIZ.csv").unlink(missing_ok=True)
        
        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"

        assert len(data_dict) == 2
        assert len(error_local) == 1
        assert len(error_yahoo) == 0
        
    def test_all_mixed_reload(self):

        backup = Path("tests/testdata/local/ZTS_BACKUP.csv")
        zts = Path("tests/testdata/local/ZTS.csv")
        shutil.copy(backup, zts)
        data_loader = data.RealDataLoader(download_delay = 0)
        data_dict, error_local, error_yahoo = data_loader._get_all_data(["XOM", "ZTS"], data_path="tests/testdata/local", update_all=True)
        zts.unlink(missing_ok=True)

        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"

        assert len(data_dict) == 2
        assert len(error_local) == 0
        assert len(error_yahoo) == 0
        
        assert data_dict["ZTS"].shape[0] >= 10 # the backup only contains 5 values


class Test_Merge_Symbols:

    def merge_two_test(self):
        vz = pd.read_csv("tests/testdata/local/VZ.csv", index_col="Date")
        wab = pd.read_csv("tests/testdata/local/WAB.csv", index_col="Date")
        wat = pd.read_csv("tests/testdata/local/WAT.csv", index_col="Date")
        data_dict = {
            "VZ" : vz,
            "WAB" : wab,
            "WAT" : wat,
            }
        data_ = data.RealDataLoader()._merge_columns(column_name="Adj Close", data_dict = data_dict)
        assert tuple(data_.shape) == (9,3)
        assert (data_.columns == ["VZ", "WAB", "WAT"]).all()
        assert pd.isna(data_.iloc[0,2])
        assert pd.isna(data_.iloc[6,0])
        assert pd.isna(data_.iloc[5,0])
        assert pd.isna(data_.iloc[4,0])
        assert pd.isna(data_.iloc[3,0])
        assert pd.isna(data_.iloc[1,2])

        # data_dict, error_dict = data_loader._get_local_data(["ABBV", "ABNB", "ABT", "ACGL"], "tests/data/local", required_columns=["Adj Close", "Close"])
    
Test_Merge_Symbols().merge_two_test()

