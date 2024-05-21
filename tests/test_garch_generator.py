from pathlib import Path

import garch_index_generator
import garch_univar_generator
import real_data_loader as data


class TestGarchGenerator:
    """Test Function to get local data"""

    def test_fit_model(self):
        data_loader = data.RealDataLoader()
        data_dict, error_dict = data_loader._get_local_data(
            ["SPY", "WAB"],
            "tests/testdata/local",
            required_columns=["Adj Close", "Close", "Open"],
        )

        assert len(data_dict) == 2
        assert error_dict == {}

        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"
        assert len(data_dict["SPY"]) == 7828
        assert set(data_dict["SPY"].columns) == set(
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

        # fit close dates
        data_ = data_dict["SPY"].loc[:, "Close"]
        generator = garch_univar_generator.GarchUnivarGenerator()
        config = {"p": 1, "q": 1, "dist": "normal"}
        model_config = generator.fit_model(data_, config)
        price, returns = generator.sample(100, 20, model_config)

        assert price.shape[0] == 101
        assert returns.shape[0] == 100

    def test_student(self):
        data_loader = data.RealDataLoader()
        data_dict, error_dict = data_loader._get_local_data(
            ["SPY", "WAB"],
            "tests/testdata/local",
            required_columns=["Adj Close", "Close", "Open"],
        )

        assert len(data_dict) == 2
        assert error_dict == {}

        for k in data_dict.keys():
            assert data_dict[k].index.name == "Date"
        assert len(data_dict["SPY"]) == 7828
        assert set(data_dict["SPY"].columns) == set(
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

        # fit close dates
        data_ = data_dict["SPY"].loc[:, "Close"]
        generator = garch_univar_generator.GarchUnivarGenerator()
        config = {"p": 4, "q": 2, "dist": "studentt"}
        model_config = generator.fit_model(data_, config)
        price, returns = generator.sample(100, 20, model_config)

        assert price.shape[0] == 101
        assert returns.shape[0] == 100

    def test_index_simple(self):
        data_loader = data.RealDataLoader(cache="tests/testdata/cache")
        real_data = data_loader.get_timeseries(data_path="tests/testdata/local")
        real_data = real_data.loc[:, ["AMZN", "GOOG", "WAB"]]

        generator = garch_index_generator.GarchIndexGenerator()
        garch_config = {"q": 2, "p": 2, "dist": "normal"}
        training_config = {"garch_config": garch_config}
        metadata = generator.fit(
            real_data,
            training_conifg=training_config,
            cache="tests/testdata/cache",
            seed=10,
        )
        samples = generator.sample(
            n_samples=100, metadata=metadata, cache="tests/testdata/cache", seed=10
        )

        assert samples[0].shape[0] == 101
        assert samples[0].shape[1] == 3
        assert set(samples[0].columns) == {"AMZN", "GOOG", "WAB"}
        assert samples[1].shape[0] == 100

        for p in Path("tests/testdata/cache").glob("*.pt"):
            p.unlink()


if __name__ == "__main__":
    TestGarchGenerator().test_index_simple()
