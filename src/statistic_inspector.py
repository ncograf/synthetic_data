import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from matplotlib.gridspec import GridSpec
from getch import getch
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
import shutil
import stylized_fact
from tueplots import bundles
import time
from tick import Tick
from copy import deepcopy

class StatisticInspector:
    """Class to clean data, visualize it and compute basic statistics
    for cleaning and augmentation purposes
    """
    
    def __init__(self, cache : str = "data/cache"):
        """Provide some tools to investigate the data

        Args:
            cache (str, optional): Directory to find cached files. Defaults to "data/cache".
        """

        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams["text.usetex"] = True

    def plot_average_sylized_fact(
            self,
            stylized_fact : stylized_fact.StylizedFact,
            rc_params : Optional[Dict[str, any]] = None, 
            copy : Optional[Path] = None, 
            ):
        """Plots a histogram of the given samples statistic

        Args:
            statistic (base_statistic.BaseStatistic): statistic to be plotted
            symbol (str): stock symbol to plot histogram of
            rc_params (Optional[Dict[str, any]], optional): matplotlib parameter for figure. Defaults to ICML 2024.
            copy (Optional[Path]): path to copy the file to other than the cache directory
            density (bool, optional): Whether to show the density instead of the values. Defaults to True.
        """
        if rc_params is None:
            rc_params = bundles.icml2024()
        
        fig_path = self.cache / f"{stylized_fact._figure_name}_average.png"

        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots()
            style : Dict[str, any] = {
                'alpha' : 1,
                'marker' : 'o',
                'markersize' : 1,
                'linestyle' : 'None'
            }
            stylized_fact.draw_stylized_fact_averaged(ax, style=style)
            fig.savefig(fig_path)
        
        if not copy is None:
            copy_path = copy / f"{stylized_fact._figure_name}_average.png"
            shutil.copy(fig_path, copy_path)

        plt.show()
