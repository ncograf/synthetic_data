from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from copulas import multivariate


class CopulaGarchModel:
    def __init__(
        self,
        copula_dict: Dict[str, any],
        garch_dict: Dict[str, any],
        initial_price: npt.ArrayLike,
    ):
        """Initilaize Copula garch

        Args:
            copula_dict (Dict[str, any]): Dictonary to create copula with the copula model.
                Should be generated with copula.to_dict().
            garch_dict (Dict[str, any]): Dictionary containing univariate garch fits. In
                particular the following keys:
                    "mu" : Dict of mean values of the fits with symbols as keys
                    "omega" : Dict of omega  values of the fits with symbols as keys
                    "alpha" : Dict of Lists with alpha parameters increasing ([alpha[1], alpha[2], ...])
                    "beta" : Dict of Lists with beta parameters increasing ([beta[1], beta[2], ...])
                Note that all alphas in the dict are supposed to have the same length
                and all the betas in the dict must have the same length as well
            initial_price (npt.ArrayLike): inital prices

        Raises:
            ValueError: If the garch_dict does not match the copula
        """

        self.copula = multivariate.GaussianMultivariate.from_dict(copula_dict)
        self.copula_dict = copula_dict
        self.garch_dict = garch_dict

        if not set(self.copula.columns) == set(garch_dict["mu"].keys()):
            raise ValueError("Given copula does not match garch_dict.")

        # extract variables to get the right columns
        self.mu = np.array([garch_dict["mu"][col] for col in self.copula.columns])
        self.omega = np.array([garch_dict["omega"][col] for col in self.copula.columns])

        # assume same univariate setting in garch i.e. same p for all symbols (also same q)
        self.alpha = np.array([garch_dict["alpha"][col] for col in self.copula.columns])
        self.beta = np.array([garch_dict["beta"][col] for col in self.copula.columns])

        self.initial_price = initial_price

    def get_model_info(self) -> Dict[str, any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "copula_dict": self.copula_dict,
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
            dtype (npt.DTypeLike, optional): Dtype for sampling. Defaults to np.float64

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price simulations and return simulations
        """

        copula_sample = np.array(self.copula.sample(length + burn))
        cols = self.copula.columns

        sigma_simulations = np.zeros((length + burn, len(cols)), dtype=dtype)
        return_simulation = np.zeros((length + burn, len(cols)), dtype=dtype)
        price_simulation = np.zeros((length + 1, len(cols)), dtype=dtype)

        sigma_simulations[0] = self.omega[0] / (
            1 - (self.alpha[:, 0] + self.beta[:, 0])
        )
        return_simulation[0] = sigma_simulations[0] * copula_sample[0]
        price_simulation[0] = self.initial_price

        def compute_sigma(i: int, sigmas: npt.NDArray, returns: npt.NDArray):
            """Compute variance for a given timestep, variances and epsilon

            Args:
                i (int): index for which to compute the variance
                sigmas (npt.NDArray): (past) standard deviations (i x D)
                returns (npt.NDArray): (past) return series (i x D)
            """

            assert i > 0

            # start with lower variance
            n_alpha = np.minimum(self.alpha.shape[1], i)
            n_beta = np.minimum(self.beta.shape[1], i)

            # alpha: D x p, beta: D x q
            variance = (
                self.omega
                + np.einsum(
                    "kl,lk->k",
                    self.alpha[:, :n_alpha],
                    returns[i - n_alpha : i, :] ** 2,
                )
                + np.einsum(
                    "kl,lk->k",
                    self.beta[:, :n_beta],
                    sigmas[i - n_beta : i, :] ** 2,
                )
            )

            return np.sqrt(variance)

        # apply garch
        for i in range(1, burn + length):
            z = copula_sample[i, :]
            sigma_simulations[i, :] = compute_sigma(
                i, sigma_simulations, return_simulation
            )
            return_simulation[i, :] = z * sigma_simulations[i, :]

        return_simulation = return_simulation[burn:] + self.mu

        # compute prices
        for i in range(1, length + 1):
            price_simulation[i] = price_simulation[i - 1] * return_simulation[i - 1]

        return price_simulation, return_simulation
