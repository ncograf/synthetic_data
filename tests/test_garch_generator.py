import garch_generator
import real_data_loader as data


class TestGarchGenerator:
    """Test Function to get local data"""

    def test_fit_model(self):
        data_loader = data.RealDataLoader()
        data_dict, error_dict = data_loader._get_local_data(
            ["SPY"],
            "tests/testdata/local",
            required_columns=["Adj Close", "Close", "Open"],
        )

        assert len(data_dict) == 1
        assert error_dict == {}

        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"
        assert len(data_dict["SPY"]) == 7828
        assert set(data_dict["SPY"].columns) == set(
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

        # fit close dates
        data_ = data_dict["SPY"].loc[:, "Close"]
        model = garch_generator.GarchGenerator()
        model.fit_model(data_)

        try:
            model.check_model()
        except:  # noqa E722
            assert False, "No error should be raised as the model is set"

        gen_price, gen_ret = model.generate_data(600, 200)
        assert gen_price.shape[0] == 600
        assert gen_ret.shape[0] == 600


if __name__ == "__main__":
    TestGarchGenerator().test_fit_model()
