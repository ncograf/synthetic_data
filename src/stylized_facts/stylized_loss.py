import coarse_fine_volatility
import leverage_effect
import linear_unpredictability
import torch
import volatility_clustering


def lu_loss(syn: torch.Tensor, real: torch.Tensor):
    n = 100

    syn_lu = torch.mean(linear_unpredictability.linear_unpredictability(syn, n), dim=1)
    real_lu_data = linear_unpredictability.linear_unpredictability(real, n)
    real_lu = torch.mean(real_lu_data, dim=1)
    real_std = torch.std(real_lu_data, dim=1)

    w = torch.exp(-torch.linspace(1, 4, real_lu.shape[0]))
    w = w / torch.sum(w)
    w = w.to(syn.device)
    return torch.sum(((syn_lu - real_lu) / real_std) ** 2 * w)


def vc_loss(syn: torch.Tensor, real: torch.Tensor):
    n = 150

    syn_vc = torch.mean(volatility_clustering.volatility_clustering(syn, n), dim=1)
    real_vc_data = volatility_clustering.volatility_clustering(real, n)
    real_vc = torch.mean(real_vc_data, dim=1)
    real_std = torch.std(real_vc_data, dim=1)

    w = torch.exp(-torch.linspace(1, 4, real_vc.shape[0]))
    w = w / torch.sum(w)
    w = w.to(syn.device)

    return torch.sum(((syn_vc - real_vc) / real_std) ** 2 * w)


def le_loss(syn: torch.Tensor, real: torch.Tensor):
    n = 100

    syn_le = torch.mean(leverage_effect.leverage_effect(syn, max_lag=n), dim=1)
    real_le_data = leverage_effect.leverage_effect(real, max_lag=n)
    real_le = torch.mean(real_le_data, dim=1)
    real_std = torch.std(real_le_data, dim=1)

    w = torch.exp(-torch.linspace(1, 4, real_le.shape[0]))
    w = w / torch.sum(w)
    w = w.to(syn.device)

    return torch.sum(((syn_le - real_le) / real_std) ** 2 * w)


def cf_loss(syn: torch.Tensor, real: torch.Tensor):
    n = 50

    syn_cf = torch.mean(
        coarse_fine_volatility.coarse_fine_volatility(syn, max_lag=n, tau=5)[0], dim=1
    )
    real_cf_data = coarse_fine_volatility.coarse_fine_volatility(
        real, max_lag=n, tau=5
    )[0]
    real_cf = torch.mean(real_cf_data, dim=1)
    real_std = torch.std(real_cf_data, dim=1)

    w = torch.exp(-torch.linspace(1, 4, real_cf.shape[0]))
    w = w / torch.sum(w)
    w = w.to(syn.device)

    return torch.sum(((syn_cf - real_cf) / real_std) ** 2 * w)


if __name__ == "__main__":
    import load_data

    sp500 = load_data.load_log_returns("sp500")
    data_a = torch.tensor(sp500[:2048, :10])
    data_b = torch.tensor(sp500[:2048, 10:20])
    c = torch.ones_like(data_b, requires_grad=True)

    loss = cf_loss(data_a, data_b * c)
    loss += le_loss(data_a, data_b * c)
    loss += lu_loss(data_a, data_b * c)
    loss += vc_loss(data_a, data_b * c)
    loss.backward()

    print(loss)
