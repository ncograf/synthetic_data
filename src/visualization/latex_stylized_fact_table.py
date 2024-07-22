from pathlib import Path
from typing import List

import load_data
import numpy.typing as npt
import stylized_score


def stylized_fact_stat(
    log_returns: npt.ArrayLike, caption: str, label: str, desc_paragraphs: List[str]
):
    stats = stylized_score._compute_stats(log_returns, "real")

    all_stat = {}
    name = ["val", "max", "std", "min", "mean", "median"]

    sel = ["mse", "mse_min", "mse_std", "mse_max", "mse_mean", "mse_median"]
    lu_stats = stats["lu_real_stat"]
    all_stat[r"linear unpredictability $\times 10^{-4}$"] = {
        ns: 10000 * lu_stats[s] for s, ns in zip(sel, name)
    }

    sel = [
        "neg_beta",
        "neg_beta_max",
        "neg_beta_std",
        "neg_beta_min",
        "neg_beta_mean",
        "neg_beta_median",
    ]
    ht_stats = stats["ht_real_stat"]
    all_stat[r"negative heavy tails"] = {ns: -ht_stats[s] for s, ns in zip(sel, name)}

    sel = [
        "pos_beta",
        "pos_beta_max",
        "pos_beta_std",
        "pos_beta_min",
        "pos_beta_mean",
        "pos_beta_median",
    ]
    ht_stats = stats["ht_real_stat"]
    all_stat[r"positive heavy tails"] = {ns: -ht_stats[s] for s, ns in zip(sel, name)}

    sel = ["beta", "beta_max", "beta_std", "beta_min", "beta_mean", "beta_median"]
    vc_stats = stats["vc_real_stat"]
    all_stat[r"volatility clustering"] = {ns: -vc_stats[s] for s, ns in zip(sel, name)}

    sel = ["beta", "beta_max", "beta_std", "beta_min", "beta_mean", "beta_median"]
    le_stats = stats["le_real_stat"]
    all_stat[r"leverage effect"] = {ns: -le_stats[s] for s, ns in zip(sel, name)}

    sel = ["beta", "beta_max", "beta_std", "beta_min", "beta_mean", "beta_median"]
    cf_stats = stats["cfv_real_stat"]
    all_stat["coarse-fine volatility"] = {ns: -cf_stats[s] for s, ns in zip(sel, name)}

    sel = [
        "arg_diff",
        "arg_diff_min",
        "arg_diff_std",
        "arg_diff_max",
        "arg_diff_mean",
        "arg_diff_median",
    ]
    gl_stats = stats["gl_real_stat"]
    all_stat[r"gain-loss asymmetry"] = {ns: gl_stats[s] for s, ns in zip(sel, name)}

    pre = (
        r"""\begin{table}
    \centering
    \begin{threeparttable}
    \caption{"""
        + caption
        + r"""}
    \label{"""
        + label
        + r"""}
    \begin{tabular}{l|l|l|l|l|l}
        \toprule
        Statistic $f$ & $s^{f}_{r}$ & Mean & Std & Min & Max\\
        """
    )
    name = ["val", "mean", "std", "max", "min"]

    post = r"""\bottomrule
    \end{tabular}
    \begin{tablenotes}[flushleft]
        \footnotesize
"""

    for d in desc_paragraphs:
        post += r"        \item " + d + "\n"

    post += r"""      \end{tablenotes}
    \end{threeparttable}
\end{table}"""

    out_str = pre
    for key in all_stat:
        out_str += r"\hline" + "\n        "
        out_str += f"{key} "
        for n in name:
            if n == "std":
                out_str += f"& ${abs(all_stat[key][n]):.2f}$ "
            else:
                out_str += f"& ${all_stat[key][n]:.2f}$ "
        out_str += r"\\" + "\n        "

    out_str += post

    return out_str


# SMI
label = "tab:smi_stf"
caption = "Empirical Stylized Facts for SMI data"
min_n = 4096

smi_prices = load_data.load_prices(index="smi")
first_date = smi_prices.index[-min_n].to_pydatetime()
last_date = smi_prices.index[-1].to_pydatetime()

smi_log_returns = load_data.get_log_returns(smi_prices, min_n)

desc = [
    f"""Quantified stylized facts computed on
        the SMI daily market data. Only including
        the data of the {smi_log_returns.shape[1]} symbols with more than~{min_n}
        successive data points from the {first_date.strftime('%b %d, %Y')}
        to the {last_date.strftime('%b %d, %Y')}.""",
    """The Value represents the estimator of the statistic
        computed on all the data or averaged data before computing
        the estimator. The remaining for values for each stylized
        fact are statistics computed over the one stock fits.""",
]

table = stylized_fact_stat(smi_log_returns, caption, label, desc)
path = Path("/home/nico/thesis/thesis/tables/smi_stf.tex")
with path.open("w") as ofile:
    ofile.write(table)


# SP500
label = "tab:sp500_stf"
caption = r"Empirical Stylized Facts for S\&P 500 data"
min_n = 4096

prices = load_data.load_prices(index="sp500")
first_date = smi_prices.index[-min_n].to_pydatetime()
last_date = smi_prices.index[-1].to_pydatetime()

log_returns = load_data.get_log_returns(prices, min_n)

desc = [
    f"""Quantified stylized facts computed on
        the S\\&P 500 daily market data. Only including
        the data of the {log_returns.shape[1]} symbols with more than~{min_n}
        successive data points from {first_date.strftime('%b %d, %Y')}
        to {last_date.strftime('%b %d, %Y')}.""",
    """The Value represents the estimator of the statistic
        computed on all the data or averaged data before computing
        the estimator. The remaining for values for each stylized
        fact are statistics computed over the one stock fits.""",
]

table = stylized_fact_stat(log_returns, caption, label, desc)
path = Path("/home/nico/thesis/thesis/tables/sp500_stf.tex")
with path.open("w") as ofile:
    ofile.write(table)

# DAX
label = "tab:dax_stf"
caption = r"Empirical Stylized Facts for DAX data"
min_n = 4096

prices = load_data.load_prices(index="dax")
first_date = smi_prices.index[-min_n].to_pydatetime()
last_date = smi_prices.index[-1].to_pydatetime()

log_returns = load_data.get_log_returns(prices, min_n)

desc = [
    f"""Quantified stylized facts computed on
        the DAX daily market data. Only including
        the data of the {log_returns.shape[1]} symbols with more than~{min_n}
        successive data points from the {first_date.strftime('%b %d, %Y')}
        to the {last_date.strftime('%b %d, %Y')}.""",
    """The Value represents the estimator of the statistic
        computed on all the data or averaged data before computing
        the estimator. The remaining for values for each stylized
        fact are statistics computed over the one stock fits.""",
]

table = stylized_fact_stat(log_returns, caption, label, desc)
path = Path("/home/nico/thesis/thesis/tables/dax_stf.tex")
with path.open("w") as ofile:
    ofile.write(table)
