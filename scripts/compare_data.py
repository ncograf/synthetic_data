import click
import real_data_loader
import garch_generator
import stock_price_statistic
import log_return_statistic
import scaled_log_return_statistic
import gen_data_loader
import tail_statistics
import distribution_comparator
import plotter
import illiquidity_filter
from arch.univariate import StudentsT
import infty_filter


@click.group()
def inspect():
    pass

@inspect.command()
def compare_garch():

    data_loader = real_data_loader.RealDataLoader()
    real_stock_data = data_loader.get_timeseries(col_name="Adj Close", data_path="data/raw_yahoo_data", update_all=False)

    garch = garch_generator.GarchGenerator(p=3,q=3,distribution=StudentsT() ,name='GARCH_3_3_student')
    #garch = garch_generator.GarchGenerator(p=1,q=1,distribution=Normal() ,name='GARCH_1_1_normal')
    gen_loader = gen_data_loader.GenDataLoader()
    synth_data = gen_loader.get_timeseries(garch, data_loader=data_loader, col_name='Adj Close')

    iliq_filter = illiquidity_filter.IlliquidityFilter()
    iliq_filter.fit_filter(real_stock_data)
    inf_filter = infty_filter.InftyFilter()
    inf_filter.fit_filter(synth_data)

    #iliq_filter.apply_filter(real_stock_data)
    #iliq_filter.apply_filter(synth_data)
    inf_filter.apply_filter(real_stock_data)
    inf_filter.apply_filter(synth_data)

    symbol = 'TSLA'
    real_stock_data = real_stock_data.loc[:,[symbol]]
    synth_data = synth_data.loc[:,[symbol]]
    
    # apply 
    
    quantile = 0.01
    real_data = stock_price_statistic.StockPriceStatistic(quantile=quantile)
    real_data.set_statistics(real_stock_data)
    real_log = log_return_statistic.LogReturnStatistic(quantile=quantile)
    real_log.set_statistics(real_stock_data)
    real_scaled_log = scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=100)
    real_scaled_log.set_statistics(real_stock_data)
    real_tail_log_upper = tail_statistics.UpperTailStatistic(real_log)
    real_tail_log_upper.set_statistics(None)
    real_tail_log_lower = tail_statistics.LowerTailStatistic(real_log)
    real_tail_log_lower.set_statistics(None)
    real_tail_scaled_upper = tail_statistics.UpperTailStatistic(real_scaled_log)
    real_tail_scaled_upper.set_statistics(None)
    real_tail_scaled_lower = tail_statistics.LowerTailStatistic(real_scaled_log)
    real_tail_scaled_lower.set_statistics(None)

    synthetic_data = stock_price_statistic.StockPriceStatistic(quantile=quantile, legend_postfix=" - GARCH")
    synthetic_data.set_statistics(synth_data)
    synthetic_log = log_return_statistic.LogReturnStatistic(quantile=quantile, legend_postfix=" - GARCH")
    synthetic_log.set_statistics(synth_data)
    syn_scaled_log = scaled_log_return_statistic.ScaledLogReturnStatistic(quantile=quantile, window=100)
    syn_scaled_log.set_statistics(synth_data)
    syn_tail_scaled_upper = tail_statistics.UpperTailStatistic(syn_scaled_log)
    syn_tail_scaled_upper.set_statistics(None)
    syn_tail_scaled_lower = tail_statistics.LowerTailStatistic(syn_scaled_log)
    syn_tail_scaled_lower.set_statistics(None)
    syn_tail_log_upper = tail_statistics.UpperTailStatistic(synthetic_log)
    syn_tail_log_upper.set_statistics(None)
    syn_tail_log_lower = tail_statistics.LowerTailStatistic(synthetic_log)
    syn_tail_log_lower.set_statistics(None)

    plot = plotter.Plotter(
        cache='data/cache',
        export='../protocol/figures',
        figure_name='distribution_divergence_' + garch.name + '_' + symbol,
        figure_title='Distribution Divergences ' + garch.name + ' ' + symbol ,
        figure_style = 
        {
            "figure.figsize" : (24,20),
            "figure.titlesize" : 24,
            "axes.titlesize" : 20,
            "axes.labelsize" : 18,
            "font.size" : 18,
            "xtick.labelsize" : 16,
            "ytick.labelsize" : 16,
            "figure.dpi" : 96,
            "figure.constrained_layout.use" : True,
            "figure.constrained_layout.h_pad" : 0.1,
            "figure.constrained_layout.hspace" : 0,
            "figure.constrained_layout.w_pad" : 0.1,
            "figure.constrained_layout.wspace" : 0,
        },
        subplot_layout={
            'ncols' : 2,
            'nrows' : 3,
            'sharex' : 'none',
            'sharey' : 'none',
        })

    tail_comp_lower = distribution_comparator.DistributionComparator(
        real_stat=real_tail_log_lower,
        syn_stat=syn_tail_log_lower,
        ax_style = {
            'title' : 'Lower Tail Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    tail_comp_lower.draw_distributions(plot.axes[0,0])

    tail_comp_upper = distribution_comparator.DistributionComparator(
        real_stat=real_tail_log_upper,
        syn_stat=syn_tail_log_upper,
        ax_style = {
            'title' : 'Upper Tail Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    tail_comp_upper.draw_distributions(plot.axes[0,1])

    tail_comp_lower = distribution_comparator.DistributionComparator(
        real_stat=real_tail_scaled_lower,
        syn_stat=syn_tail_scaled_lower,
        ax_style = {
            'title' : 'Lower Tail Scaled Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    tail_comp_lower.draw_distributions(plot.axes[1,0])

    tail_comp_upper = distribution_comparator.DistributionComparator(
        real_stat=real_tail_scaled_upper,
        syn_stat=syn_tail_scaled_upper,
        ax_style = {
            'title' : 'Upper Tail Scaled Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    tail_comp_upper.draw_distributions(plot.axes[1,1])

    log_comp = distribution_comparator.DistributionComparator(
        real_stat=real_log,
        syn_stat=synthetic_log,
        ax_style = {
            'title' : 'Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    log_comp.draw_distributions(plot.axes[2,0])

    scaled_log_comp = distribution_comparator.DistributionComparator(
        real_stat=real_scaled_log,
        syn_stat=syn_scaled_log,
        ax_style = {
            'title' : 'Scaled Log Return',
            'ylabel' : r'Density',
            'xlabel' : r'scaled log returns',
            'xscale' : 'linear',
            'yscale' : 'log',
            }
    )
    scaled_log_comp.draw_distributions(plot.axes[2,1])
    plot.save()
    
if __name__ == "__main__":
    compare_garch()