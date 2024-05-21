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
            nu = garch_dict["nu"]
            self.sampler = lambda n: scipy.stats.t.rvs(df=nu, size=n)
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

        sigma_simulations = np.zeros((length + burn), dtype=dtype)
        return_simulation = np.zeros((length + burn), dtype=dtype)
        price_simulation = np.zeros((length + 1), dtype=dtype)

        sigma_simulations[0] = self.omega / (1 - (self.alpha[0] + self.beta[0]))
        return_simulation[0] = sigma_simulations[0] * samples[0]
        price_simulation[0] = self.initial_price

        def compute_sigma(i: int, sigmas: npt.NDArray, returns: npt.NDArray):
            """Compute variance for a given timestep, variances and epsilon

            Args:
                i (int): index for which to compute the variance
                sigmas (npt.NDArray): (past) standard deviation of size at least i
                returns (npt.NDArray): (past) return series of size at least i
            """

            assert i > 0

            # start with lower variance
            n_alpha = np.minimum(self.alpha.shape[0], i)
            n_beta = np.minimum(self.beta.shape[0], i)

            # alpha: p, beta: q
            variance = (
                self.omega
                + np.dot(self.alpha[:n_alpha], returns[i - n_alpha : i] ** 2)
                + np.dot(self.beta[:n_beta], sigmas[i - n_beta : i] ** 2)
            )

            return np.sqrt(variance)

        # apply garch
        for i in range(1, burn + length):
            z = samples[i]
            sigma_simulations[i] = compute_sigma(
                i, sigma_simulations, return_simulation
            )
            return_simulation[i] = z * sigma_simulations[i]

        return_simulation = return_simulation[burn:] + self.mu

        # compute prices
        for i in range(1, length + 1):
            price_simulation[i] = price_simulation[i - 1] * return_simulation[i - 1]

        return price_simulation, return_simulation
