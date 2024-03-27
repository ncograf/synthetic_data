import click
import numpy as np
import real_data_loader
import garch_generator
import stock_price_statistic
import log_return_statistic
import index_generator
import scipy.stats as ss
import scipy.spatial.distance as sd


@click.group()
def inspect():
    pass

@inspect.command()
def compare_garch():


    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)
    real_stock_data = real_stock_data.loc[:,real_stock_data.columns[:10]]

    garch = garch_generator.GarchGenerator()
    index_gen = index_generator.IndexGenerator(generator=garch)
    synth_data = index_gen.generate_index(real_stock_data)
    
    real_data = stock_price_statistic.StockPriceStatistic(quantile=0.1)
    real_data.set_statistics(real_stock_data)
    real_log = log_return_statistic.LogReturnStatistic(quantile=0.1)
    real_log.set_statistics(real_stock_data)

    synthetic_data = stock_price_statistic.StockPriceStatistic(quantile=0.1, legend_postfix=" - GARCH")
    synthetic_data.set_statistics(synth_data)
    synthetic_log = log_return_statistic.LogReturnStatistic(quantile=0.1, legend_postfix=" - GARCH")
    synthetic_log.set_statistics(synth_data)

    figure_params = {"figure.figsize" : (16, 10),
             "font.size" : 24,
             "figure.dpi" : 96,
             "figure.constrained_layout.use" : True,
             "figure.constrained_layout.h_pad" : 0.1,
             "figure.constrained_layout.hspace" : 0,
             "figure.constrained_layout.w_pad" : 0.1,
             "figure.constrained_layout.wspace" : 0,
            }

    syn_kde = synthetic_log.get_kde_all()
    rel_kde = real_log.get_kde_all()
    
    rel_data = real_log.statistic
    rel_data = rel_data[~np.isnan(rel_data)]
    syn_data = synthetic_log.statistic
    syn_data = syn_data[~np.isnan(syn_data)]
    
    max = np.nanmax(rel_data)
    min = np.nanmin(rel_data)
    dist = max - min
    space = np.linspace(min - dist / 2, max + dist / 2, 1000)
    pdf_syn = syn_kde.pdf(space)
    pdf_rel = rel_kde.pdf(space)
    js_dist = sd.jensenshannon(pdf_rel, pdf_syn)
    w_dist = ss.wasserstein_distance(rel_data,syn_data)
    print("Wasserstein distance ", w_dist)
    print("Jenson Shennon distance ", js_dist)

    
if __name__ == "__main__":
    compare_garch()