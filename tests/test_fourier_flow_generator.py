import os

import fourier_flow_generator
import real_data_loader as data


class TestFourierFlowGenerator:
    """Test Function to get local data"""

    def test_fit_model(self):
        os.environ["WANDB_MODE"] = "disabled"
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
        data_ = data_.iloc[:100]
        model = fourier_flow_generator.FourierFlowGenerator(symbol="MSFT")
        config = {
            "hidden_dim": 20,
            "num_layer": 2,
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.001,
            "gamma": 0.999,
            "seq_len": 11,
            "lag": 1,
        }
        model.fit_model(price_data=data_, **config)

        try:
            model.check_model()
        except:  # noqa E722
            assert False, "No error should be raised as the model is set"

        gen_price, gen_ret = model.generate_data(
            model=model.model(),
            scale=model.data_amplitude,
            shift=model.data_min,
            init_price=model._zero_price,
            len_=20,
        )

        assert gen_price.shape[0] == 20
        assert gen_ret.shape[0] == 20


if __name__ == "__main__":
    TestFourierFlowGenerator().test_fit_model()
