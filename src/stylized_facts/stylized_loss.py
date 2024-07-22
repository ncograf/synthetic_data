import coarse_fine_volatility
import leverage_effect
import linear_unpredictability
import torch
import volatility_clustering


def lu_loss(syn: torch.Tensor):
    lin_upred = linear_unpredictability.linear_unpredictability_torch(syn, 1000)
    return torch.mean(lin_upred**2)


def vc_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_vc = torch.mean(
        volatility_clustering.volatility_clustering_torch(syn, 500), dim=1
    )
    real_vc = torch.mean(
        volatility_clustering.volatility_clustering_torch(real, 500), dim=1
    )

    return torch.mean((syn_vc - real_vc) ** 2)


def le_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_le = torch.mean(leverage_effect.leverage_effect_torch(syn, max_lag=100), dim=1)
    real_le = torch.mean(
        leverage_effect.leverage_effect_torch(real, max_lag=100), dim=1
    )

    return torch.mean((syn_le - real_le) ** 2)


def cf_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_le = torch.mean(
        coarse_fine_volatility.coarse_fine_volatility_torch(syn, max_lag=100)[0], dim=1
    )
    real_le = torch.mean(
        coarse_fine_volatility.coarse_fine_volatility_torch(real, max_lag=100)[0], dim=1
    )

    return torch.mean((syn_le - real_le) ** 2)
