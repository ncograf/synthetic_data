import numpy as np
import numpy.typing as npt
import scipy.optimize as optim
import torch
from scipy.special import gamma
from torch.func import jacrev


def fit_gen_inv_gamma(
    disc_pdf_data: npt.ArrayLike, theta_0: npt.ArrayLike, method: str
) -> npt.NDArray:
    """Fit Max Likelyhood of generalized inverse gamma distribution

    Args:
        disc_pdf_data (npt.ArrayLike): discrete pdf data (disc_pdf_data[i] := #t s.t. inf[t' | t' > rho] = t)
        theta_0 (npt.ArrayLike): initial parameters [t_0, nu, alpha, beta]
        method (str): scipy optimization method (L-BFGS-B, BFGS, Newton-CG, ...).

    Returns:
        npt.NDArray: [t_0, nu, alpha, beta]
    """

    # pytorch data
    freq_t = torch.from_numpy(np.array(disc_pdf_data))
    t = torch.arange(1, freq_t.numel() + 1, dtype=freq_t.dtype)

    # negative log likelihood for generalized inverse gamma
    def neg_log_like(theta: torch.Tensor) -> torch.Tensor:
        """Compute the log pdf of the generalized inverse gamma distribution

        Args:
            theta (torch.Tensor): [t_0, nu, alpha, beta]

        Returns:
            torch.Tensor: negative log likelyhood over the data
        """

        t_0 = theta[0]
        nu = theta[1]
        alpha = theta[2]
        beta = theta[3]

        sum_t = t + t_0

        first_term = torch.log(nu) - torch.lgamma(alpha / nu)
        second_term = torch.log(torch.abs(beta) ** (2 * alpha)) - torch.log(
            sum_t ** (alpha + 1)
        )
        third_term = -((beta**2 / sum_t) ** nu)

        # sum up likelyhoods for all t
        return -torch.dot((first_term + second_term + third_term), freq_t)

    # neg_log_likelyhood
    def np_neg_log_like(theta: npt.ArrayLike) -> npt.ArrayLike:
        theta = torch.tensor(theta)
        return neg_log_like(theta).detach().cpu().numpy()

    # jacobian function for generalized inverse gamma log likelihood
    def np_jacobian(theta: npt.ArrayLike) -> npt.NDArray:
        theta = torch.tensor(theta)
        return jacrev(neg_log_like)(theta).detach().cpu().numpy()

    # hessian for generalizd inverse gamma
    def np_hessian(theta: npt.ArrayLike) -> npt.NDArray:
        theta = torch.tensor(theta)
        return jacrev(jacrev(neg_log_like))(theta).detach().cpu().numpy()

    result: optim.OptimizeResult = optim.minimize(
        fun=np_neg_log_like, x0=theta_0, method=method, jac=np_jacobian, hess=np_hessian
    )

    if result.success:
        print(f"GenInvGamma fit converged successfully after {result.nit} iterations.")
    else:
        print(
            f"GenInvGamma fit did NOT converge successfully after {result.nit} iterations."
        )

    return result.x, np_neg_log_like(result.x)


def gen_inf_gamma_pdf(t: npt.ArrayLike, theta: npt.ArrayLike) -> npt.NDArray:
    """Compute generalized inverse gamma pdf function

    Args:
        t (npt.ArrayLike): t (evaluation points)
        theta (npt.ArrayLike): [t_0, nu, alpha, beta] function parameter

    Returns:
        ndarray: pdf evaluations at t
    """

    t_0 = theta[0]
    nu = theta[1]
    alpha = theta[2]
    beta = theta[3]

    term_one = nu / gamma(alpha / nu)
    term_two = np.abs(beta) ** (2 * alpha) / (t + t_0) ** (alpha + 1)
    term_three = np.exp(-((beta**2 / (t + t_0)) ** nu))

    return term_one * term_two * term_three


def gen_inf_gamma_max(theta: npt.ArrayLike) -> npt.NDArray:
    """Compute generalized inverse gamma maximum

    Args:
        theta (npt.ArrayLike): [t_0, nu, alpha, beta] function parameter

    Returns:
        float: max point of the function
    """

    t_0 = theta[0]
    nu = theta[1]
    alpha = theta[2]
    beta = theta[3]

    max_t = beta**2 * (nu / (alpha + 1)) ** (1 / nu) - t_0

    return max_t
