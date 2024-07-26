import coarse_fine_volatility
import leverage_effect
import linear_unpredictability
import torch
import volatility_clustering


def lu_loss(syn: torch.Tensor):
    lin_upred = linear_unpredictability.linear_unpredictability(syn, 100)
    return torch.mean(lin_upred**2)


def vc_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_vc = torch.mean(volatility_clustering.volatility_clustering(syn, 150), dim=1)
    real_vc = torch.mean(volatility_clustering.volatility_clustering(real, 150), dim=1)

    return torch.mean((syn_vc - real_vc) ** 2)


def le_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_le = torch.mean(leverage_effect.leverage_effect(syn, max_lag=80), dim=1)
    real_le = torch.mean(leverage_effect.leverage_effect(real, max_lag=80), dim=1)

    return torch.mean((syn_le - real_le) ** 2)


def cf_loss(syn: torch.Tensor, real: torch.Tensor):
    syn_le = torch.mean(
        coarse_fine_volatility.coarse_fine_volatility(syn, max_lag=50, tau=5)[0], dim=1
    )
    real_le = torch.mean(
        coarse_fine_volatility.coarse_fine_volatility(real, max_lag=50, tau=5)[0], dim=1
    )

    return torch.mean((syn_le - real_le) ** 2)


if __name__ == "__main__":
    import load_data

    sp500 = load_data.load_log_returns("sp500")
    data_a = torch.tensor(sp500[:2048, :10]).T.unsqueeze(1)
    data_b = torch.tensor(sp500[:2048, 10:20]).T.unsqueeze(1)
    c = torch.ones_like(data_b, requires_grad=True)

    loss = cf_loss(data_a, data_b * c)
    loss += le_loss(data_a, data_b * c)
    loss += lu_loss(data_b * c)
    loss += vc_loss(data_a, data_b * c)
    loss.backward()

    print(loss)
