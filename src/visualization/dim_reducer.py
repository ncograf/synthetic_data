from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimReducer:
    def __init__(self, kind: Literal["PCA", "TSNE"] = "TSNE"):
        """Initialize a dimensionality reducer with one of the available methods

        Args:
            kind (Literal['PCA';, 'TSNE'], optional): Method used for the reduction. Defaults to 'TSNE'.
        """

        self.kind = kind

    def reduce(self, X: npt.ArrayLike) -> npt.NDArray:
        """Reduce the data to two dimensions, X is supposed to contain a sample per row.
        So the output will have the same number of rows, but only two columns

        Args:
            X (npt.ArrayLike): Data to be reduced

        Returns:
            npt.NDArray: Reduced data
        """

        # Make sure to have the right dimension by taking the mean over the rest.
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], -1))
        X = np.nanmean(X, axis=-1)

        nan_mask = np.isnan(X)
        if np.sum(nan_mask) > 0:
            print(f"Dim reducer imputeded {np.sum(nan_mask)} nan values with 0")
            X[nan_mask] = 0

        if self.kind == "PCA":
            transform = PCA(n_components=2)

        if self.kind == "TSNE":
            transform = TSNE(n_components=2)

        dim_reduct = transform.fit_transform(X)

        return dim_reduct

    def draw_reduction(self, ax: plt.Axes, X: npt.ArrayLike, **kwargs):
        """Computes and draws the dimensonality reduction onto the axis

        Args:
            ax (plt.Axes): Axis to draw onto
            X (npt.ArrayLike): Data to be reduced and drawn
        """

        low_dim = self.reduce(X)
        ax.scatter(low_dim[:, 0], low_dim[:, 1], **kwargs)
        ax.legend()

    def visualize_dim_reduction(
        self,
        data_one: pd.DataFrame,
        data_two: pd.DataFrame,
    ) -> plotter.Plotter:
        """Draw dimensionality reduction for two datasets

        Args:
            data_one (pd.DataFrame): First dataset
            data_two (pd.DataFrame): Second dataset

        Returns:
            Plotter: plotter containing the dimensionality reduction
        """
        plot = plotter.Plotter(
            cache="data/cache",
            figure_style={
                "figure.figsize": (16, 10),
                "figure.titlesize": 24,
                "axes.titlesize": 20,
                "axes.labelsize": 18,
                "font.size": 18,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "figure.dpi": 96,
                "figure.constrained_layout.use": True,
                "figure.constrained_layout.h_pad": 0.1,
                "figure.constrained_layout.hspace": 0,
                "figure.constrained_layout.w_pad": 0.1,
                "figure.constrained_layout.wspace": 0,
            },
            figure_title="TSNE Projections",
            subplot_layout={
                "ncols": 1,
                "nrows": 1,
                "sharex": "all",
                "sharey": "all",
            },
        )

        ax = plot.axes

        style_one = {
            "color": "blue",
        }

        style_two = {
            "color": "red",
        }

        self.draw_reduction(ax, data_one, **style_one)
        self.draw_reduction(ax, data_two, **style_two)

        return plot
