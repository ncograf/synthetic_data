from typing import List, Tuple

import inflect
import numpy as np

p = inflect.engine()


def prepare_data(*data: List[Tuple[float, List[float], List[float], str]]):
    data_ = data

    out_data = []
    for i, data in enumerate(data_):
        data.sort(key=lambda x: x[0])

        total_scores, scores, wd, _ = zip(*data)
        tot_min = np.min(total_scores)
        min = np.min(scores, axis=0)
        tot_max = np.max(total_scores)
        max = np.max(scores, axis=0)
        tot_avg = np.mean(total_scores)
        avg = np.mean(scores, axis=0)
        wd_min = np.min(wd, axis=0)
        wd_avg = np.mean(wd, axis=0)
        wd_max = np.max(wd, axis=0)

        min_exp = data[0]
        med_exp = data[len(data) // 2]
        max_exp = data[-1]

        out_data.append((*min_exp[:3], f"MIN - {min_exp[3]}"))
        out_data.append((*med_exp[:3], f"MED - {med_exp[3]}"))
        out_data.append((*max_exp[:3], f"MAX - {max_exp[3]}"))
        out_data.append((tot_min, min, wd_min, "MIN"))
        out_data.append((tot_avg, avg, wd_avg, "AVG"))
        out_data.append((tot_max, max, wd_max, "MAX"))
        out_data.append(
            (*corr(wd, total_scores, scores), r"Corr -- $\mathcal{S} - W_1$")
        )
        # out_data.append((*corr(np.log(wd), total_scores, scores), r"Corr -- $\mathcal{S} - \log W_1$"))

        if i != len(data_):
            out_data.append("hline")

    return out_data


def corr(wd, tot, sc):
    wd_mean = np.mean(wd)
    wd_std = np.nanstd(wd)
    tot_mean = np.mean(tot)
    tot_std = np.nanstd(tot)
    sc_mean = np.mean(sc, axis=0)
    sc_std = np.nanstd(sc, axis=0)
    corr_tot = np.mean((np.asarray(wd) - wd_mean) * (np.asarray(tot) - tot_mean)) / (
        tot_std * wd_std
    )
    corr_sc = np.mean(
        (np.asarray(wd).reshape((-1, 1)) - wd_mean)
        * (np.asarray(sc) - sc_mean.reshape((1, -1))),
        axis=0,
    ) / (sc_std * wd_std)
    return corr_tot, corr_sc, [1]


# function to create a table
def create_table(
    name: str,
    fig_name: str,
    plot: str,
    caption: str,
    data: List[Tuple[float, List[float], List[float], str]],
    out_dir: str,
):
    text = f"""Stylized scores corresponding to plot {plot} in Figure~\\ref{{fig:{fig_name}}}
                The table includes the experiments for the plotted points in the Figure.
                In particular, the minimum, median, and maximum experiment measurements of the
                average stylized score. Moreover, the minimum, average, and maximum for the
                each score individually over all the experiments is listed as well."""
    table_start = f"""\\begin{{table}}
    \\begin{{adjustwidth}}{{-1in}}{{-1in}}
    \\centering
    \\begin{{threeparttable}}
        \\caption{{{caption}}}
        \\label{{tab:{name}}}
            \\begin{{tabular}}{{c|l|l|l|l|l|l|l|l}}
                \\toprule
                experiment & $\\mathcal{{S}}$
                & $\\mathcal{{S}}^{{lu}}$ & $\\mathcal{{S}}^{{ht}}$
                & $\\mathcal{{S}}^{{vc}}$ & $\\mathcal{{S}}^{{le}}$ 
                & $\\mathcal{{S}}^{{cf}}$ & $\\mathcal{{S}}^{{gl}}$
                & $W_1(R',R)$\\\\"""
    table_end = f"""            \\bottomrule
            \\end{{tabular}}
            \\begin{{tablenotes}}[flushleft]
                \\footnotesize
                \\item {text}
            \\end{{tablenotes}}
    \\end{{threeparttable}}
    \\end{{adjustwidth}}
    \\end{{table}}"""

    table = table_start + "\n"
    for dat in data:
        if dat == "hline":
            table += r"                \hline" + "\n"
        else:
            mu, scs, wd, run = dat
            ind_scores_str = " & ".join([f"${d:.2f}$" for d in scs])
            wd_scores_str = " & ".join([f"${d:.2e}$" for d in wd])
            label = run.replace("stylized loss", r"$\mathcal{S}$-loss")
            table += r"                \hline" + "\n"
            table += f"                {label} & {mu:.2f}\n"
            table += f"                & {ind_scores_str} & {wd_scores_str}\\\\\n"
    table += table_end
    with (out_dir / f"{name}.tex").open("w") as file:
        file.write(table)


def std_table(first_sentence: str, stvars, data_name, table_name):
    text = f"""{first_sentence}
    To obtain the stylized score,
    these standard deviations scale the estimators before
    comparing them with a Wasserstein distance.
    The statistics are computed over the variance estimations
    for the different lag values~$k$, where the
    the range of the relevant~$k$ depends on the stylized fact.
    """
    table_start = f"""\\begin{{table}}
    \\begin{{adjustwidth}}{{-1in}}{{-1in}}
    \\centering
    \\begin{{threeparttable}}
        \\caption{{{data_name} stylized scaling parameters}}
        \\label{{tab:{table_name}}}
            \\begin{{tabular}}{{c|l|l|l|l|l|l}}
                \\toprule
                statistic over $k$
                & $\\sigma^{{lu}}$ & $\\sigma^{{ht}}$
                & $\\sigma^{{vc}}$ & $\\sigma^{{le}}$
                & $\\sigma^{{cf}}$ & $\\sigma^{{gl}}$\\\\"""
    table_end = f"""            \\bottomrule
            \\end{{tabular}}
            \\begin{{tablenotes}}[flushleft]
                \\footnotesize
                \\item {text}
            \\end{{tablenotes}}
    \\end{{threeparttable}}
    \\end{{adjustwidth}}
    \\end{{table}}"""

    table = table_start + "\n"

    data = list(zip(*stvars))[:3]
    for vals, run in zip(data, ["AVG", "MAX", "MIN"]):
        label = run.replace("stylized loss", r"$\mathcal{S}$-loss")
        ind_scores_str = " & ".join([f"${d:.4f}$" for d in vals])
        table += r"                \hline" + "\n"
        table += f"                {label}\n"
        table += f"                & {ind_scores_str}\\\\\n"
    table += table_end

    return table
