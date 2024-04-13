import quantile_statistic as stat
import pandas as pd
import numpy as np
import matplotcheck.base as mpc
import matplotlib.pyplot as plt
from tick import Tick


class TestQuantileStatistic:
    def test_set_statistic(self):
        base_stat = stat.QuantileStatistic(quantile=0.4)
        cols = ["a", "b", "c", "d"]
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        now = pd.Timestamp.now().floor("d")
        later = pd.Timestamp("2017-01-01 12:00:00").floor("d")
        never = pd.Timestamp("2018-01-01 12:00:00").floor("d")
        indices = [now, later, never]
        base_df = pd.DataFrame(data=data, index=indices, columns=cols)

        base_stat.set_statistics(data=base_df)
        assert np.all(base_stat.statistic == base_df)

        outliers = base_stat.get_outlier()
        outlier_check = {
            Tick(now, "a"),
            Tick(never, "a"),
            Tick(now, "b"),
            Tick(never, "b"),
            Tick(now, "c"),
            Tick(never, "c"),
            Tick(now, "d"),
            Tick(never, "d"),
        }
        assert outliers == outlier_check

    def test_histogram(self):
        base_stat = stat.QuantileStatistic(quantile=0.4)
        cols = ["a", "b", "c", "d"]
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        now = pd.Timestamp.now()
        later = pd.Timestamp("2017-01-01 12:00:00")
        never = pd.Timestamp("2018-01-01 12:00:00")
        indices = [now, later, never]
        base_df = pd.DataFrame(data=data, index=indices, columns=cols)

        base_stat.set_statistics(data=base_df)

        fig, ax = plt.subplots(1, 1)
        base_stat.draw_histogram(ax, "a")
        tester = mpc.PlotTester(ax)
        tester.assert_axis_label_contains("x")
        tester.assert_axis_label_contains("y")
        tester.assert_plot_type("bar")

    def test_plot_point(self):
        base_stat = stat.QuantileStatistic(quantile=0.4)
        cols = ["a", "b", "c", "d"]
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        now = pd.Timestamp.now().floor("d")
        later = pd.Timestamp("2017-01-01 12:00:00").floor("d")
        never = pd.Timestamp("2018-01-01 12:00:00").floor("d")
        indices = [now, later, never]
        base_df = pd.DataFrame(data=data, index=indices, columns=cols)

        base_stat.set_statistics(data=base_df)

        fig, ax = plt.subplots(1, 1)
        base_stat.draw_point(ax, Tick(now, "a"))
        tester = mpc.PlotTester(ax)
        tester.assert_plot_type("line")

    def test_error(self):
        try:
            _ = stat.QuantileStatistic(quantile=2)
            assert False, "Value error expected."
        except ValueError as ex:
            assert "The quantile must be" in str(ex)

        try:
            _ = stat.QuantileStatistic(quantile=-0.1)
            assert False, "Value error expected."
        except ValueError as ex:
            assert "The quantile must be" in str(ex)
