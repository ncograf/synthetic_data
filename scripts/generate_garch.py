import garch_generator
import real_data_loader
import gen_data_loader
from arch.univariate import StudentsT, Normal
import click
import sp500_statistic


@click.command()
@click.option('-d', '--dist',
              type=click.Choice(['StudentT', 'Normal'], case_sensitive=False),
              help='Distribution for univariate Garch Sample.')
@click.option('-p', 
              type=int, 
              help='Parameter p for garch model')
@click.option('-q', 
              type=int, 
              help='Parameter q for garch model')
@click.option('-n', '--num_cpu', type=int, default=1, help='Number of CPUs to be used.')
def generate_garch_index(dist : str, p, q, num_cpu):
    """Generate garch data and return simplest summary"""

    if dist.lower() == 'studentt':
        dist_ = StudentsT()
    elif dist.lower() == 'normal':
        dist_ = Normal()
    else:
        raise ValueError("The distribution must be chosen out of the predefined options.")
    
    print(f'Distribution : {dist}')

    data_loader = real_data_loader.RealDataLoader()
    garch = garch_generator.GarchGenerator(
        p=p, q=q, distribution=dist_, name=f"GARCH_p{p}_q{q}_{dist.lower()}"
    )
    # garch = garch_generator.GarchGenerator(p=1,q=1,distribution=Normal() ,name='GARCH_1_1_normal')
    gen_loader = gen_data_loader.GenDataLoader()
    synth_data = gen_loader.get_timeseries(
        garch, data_loader=data_loader, col_name="Adj Close", n_cpu=num_cpu,
    )
    
    stat = sp500_statistic.SP500Statistic()
    stat.set_statistics(synth_data)
    stat.print_distribution_properties()

if __name__ == '__main__':
    generate_garch_index()