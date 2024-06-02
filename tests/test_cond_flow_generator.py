import os

import conditional_flow_generator
import real_data_loader as data
from accelerate import Accelerator


class TestCondFlowGenerator:
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
        data_ = data_dict["SPY"].loc[:, ["Close", "High"]]
        data_ = data_.iloc[:100]
        model = conditional_flow_generator.ConditionalFlowGenerator()
        config = {
            "cond_flow_config": {
                "hidden_dim": 20,
                "dim": 2,
                "conditional_dim": 20,
                "n_layer": 1,
                "num_model_layer": 2,
                "drop_out": 0.5,
                "norm": "layer",
                "activation": "sigmoid",
                "dtype": "float32",
            },
            "seq_len": 32,
            "batch_size": 128,
            "epochs": 10,
            "optim_config": {
                "lr": 0.001,
            },
            "lr_config": {
                "gamma": 0.999,
            },
            "lag": 1,
            "sym": "MSFT",
        }

        os.environ["WANDB_MODE"] = "disabled"
        model_dict = model.fit(
            price_data=data_, config=config, accelerator=Accelerator()
        )
        model_dict["dtype"] = "float32"
        gen_price, gen_ret = model.sample(config=model_dict, length=20, burn=0)

        assert gen_price.shape[0] == 21
        assert gen_ret.shape[0] == 20


if __name__ == "__main__":
    TestCondFlowGenerator().test_fit_model()
