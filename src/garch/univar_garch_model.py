from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import scipy
import scipy.stats


class UnivarGarchModel:
    def __init__(
        self,
        garch_dict: Dict[str, any],
        initial_price: float,
    ):
        """Initilaize univar garch

        Args:
            garch_dict (Dict[str, any]): Dictionary containing the univariate garch fit. In
                particular the following keys:
                    "mu" : mean value
                    "omega" : omega  value
                    "alpha" : List with alpha parameters increasing ([alpha[1], alpha[2], ...])
                    "beta" : List with beta parameters increasing ([beta[1], beta[2], ...])
                    "dist": distributions, Literal ['normal', 'studentT']
                    "nu" : degrees of freedom only necessary for studentT distribution
            initial_price (float): inital prices
        """
        self.garch_dict = garch_dict

        # extract variables to get the right columns
        self.mu = garch_dict["mu"]
        self.omega = garch_dict["omega"]

        # assume same univariate setting in garch i.e. same p for all symbols (also same q)
        self.alpha = np.array(garch_dict["alpha"])
        self.beta = np.array(garch_dict["beta"])

        dist = garch_dict["dist"]
        if dist.lower() == "studentt":
            self.nu = garch_dict["nu"]

            if self.nu <= 2:
                raise ValueError(
                    f"The degrees of freedom ({self.nu}) must be larger than 2 for studentT fit."
                )
            # see arch for implemtation details
            std = np.sqrt(self.nu / (self.nu - 2))
            self.sampler = lambda n: scipy.stats.t.rvs(df=self.nu, size=n) / std
        elif dist.lower() == "normal":
            self.sampler = lambda n: scipy.stats.norm.rvs(size=n)
        else:
            raise ValueError(f"Distribution {dist} not known!")

        self.initial_price = initial_price

    def get_model_info(self) -> Dict[str, any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "garch_dict": self.garch_dict,
            "initial_price": self.initial_price,
        }

        return dict_

    def sample(
        self, length: int, burn: int, dtype: npt.DTypeLike = np.float64
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from garch copula

        Args:
            length (int): Number of samples.
            burn (int): Number of samples in the beginning to be neglected (recommended q << burn, p << burn).
            dtype (npt.DTypeLike, optional): Dtype for sampling

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price simulations and return simulations
        """

        samples = self.sampler(length + burn)

        min_lag = np.maximum(len(self.beta), len(self.alpha))

        sigma_simulations = np.ones((length + burn), dtype=dtype)
        return_simulation = np.zeros((length + burn), dtype=dtype)
        price_simulation = np.zeros((length + 1), dtype=dtype)

        # initialization copied from https://github.com/bashtage/arch/blob/main/arch/univariate/volatility.py#L1138
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        sigma_simulations[:min_lag] = (
            self.omega if persistence > 1 else self.omega / (1.0 - persistence)
        )
        return_simulation[:min_lag] = (
            np.sqrt(sigma_simulations[:min_lag]) * samples[:min_lag]
        )
        price_simulation[0] = self.initial_price

        # apply garch
        n_alpha = len(self.alpha)
        n_beta = len(self.beta)
        for t in range(min_lag, length + burn):
            sigma_simulations[t] = (
                self.omega
                + np.dot(self.alpha, np.flip(return_simulation[t - n_alpha : t] ** 2))
                + np.dot(self.beta, np.flip(sigma_simulations[t - n_beta : t]))
            )
            return_simulation[t] = samples[t] * np.sqrt(sigma_simulations[t])

        return_simulation = 1 + (self.mu + return_simulation[burn:]) / 100.0

        # compute prices
        for i in range(1, length + 1):
            price_simulation[i] = price_simulation[i - 1] * return_simulation[i - 1]

        return price_simulation, return_simulation
