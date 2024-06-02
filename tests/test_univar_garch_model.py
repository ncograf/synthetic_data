import numpy as np
import univar_garch_model
from accelerate.utils import set_seed


class TestUnivarGarchModel:
    """Test Function to get local data"""

    def test_sample(self):
        model_dict = {"mu": 0, "omega": 0, "alpha": [0], "beta": [0], "dist": "normal"}
        initial_price = 0
        model = univar_garch_model.UnivarGarchModel(model_dict, initial_price)

        s = model.sample(2, 0)

        assert len(s) == 2
        assert len(s[0]) == 3
        assert len(s[1]) == 2

    def test_more_q(self):
        model_dict = {
            "mu": 0.1,
            "omega": 0.1,
            "alpha": [0, 0.1],
            "beta": [0, 0.3, 0.3],
            "dist": "normal",
        }
        initial_price = 0
        model = univar_garch_model.UnivarGarchModel(model_dict, initial_price)

        set_seed(20)
        s = model.sample(4, 0)

        assert len(s) == 2
        assert len(s[0]) == 5
        assert len(s[1]) == 4

        model_dict = {
            "mu": 0.1,
            "omega": 0.1,
            "alpha": [0, 0.1],
            "beta": [0, 0.1, 0.2],
            "dist": "normal",
        }
        model = univar_garch_model.UnivarGarchModel(model_dict, initial_price)

        set_seed(20)
        ss = model.sample(4, 0)
        assert np.all(ss[1][2:] != s[1][2:])

    def test_more_dist(self):
        set_seed(20)

        model_dict = {
            "mu": 0.1,
            "omega": 0.1,
            "alpha": [0, 0.1],
            "beta": [0, 0.3, 0.1],
            "dist": "studentt",
            "nu": 3,
        }
        initial_price = 0
        model = univar_garch_model.UnivarGarchModel(model_dict, initial_price)

        s = model.sample(4, 0)

        assert len(s) == 2
        assert len(s[0]) == 5
        assert len(s[1]) == 4


if __name__ == "__main__":
    TestUnivarGarchModel().test_more_q()
