import base_statistic as stat
import pandas as pd
import numpy as np
from tick import Tick

class TestBaseStatistic:
    
    def test_get_statistic(self):
        
        base_stat = stat.BaseStatistic()
        base_stat._symbols = ["a", "b", "c", "d"]
        base_stat._statistic = np.array([[1,2,3,4],[5,6,7,8]])
        now = pd.Timestamp.now()
        later = pd.Timestamp('2017-01-01 12:00:00')
        base_stat._dates = [now.floor('d'), later.floor('d')]
        first_tick = Tick(now, "a")
        second_tick = Tick(later, "c")
        
        assert np.all(base_stat.statistic == base_stat._statistic)
        assert base_stat.get_statistic(first_tick) == 1
        assert base_stat.get_statistic(second_tick) == 7

    def test_get_statistic_multidim(self):
        
        base_stat = stat.BaseStatistic()
        base_stat._symbols = ["a", "b", "c", "d"]
        base_stat._statistic = np.ones((2,4,3))
        now = pd.Timestamp.now().floor('d')
        later = pd.Timestamp('2017-01-01 12:00:00').floor('d')
        base_stat._dates = [now, later]
        first_tick = Tick(now, "a")
        second_tick = Tick(later, "d")
        
        assert np.all(base_stat.statistic == base_stat._statistic)
        assert np.all(base_stat.get_statistic(first_tick) == np.ones(3))
        assert np.all(base_stat.get_statistic(second_tick) == np.ones(3))

    def test_check_data_validity(self):

        base_stat = stat.BaseStatistic()
        base_stat._symbols = ["a", "b", "c", "d"]
        base_stat._statistic = np.array([[1,2,3,4],[5,6,7,8]])
        now = pd.Timestamp.now()
        later = pd.Timestamp('2017-01-01 12:00:00')
        base_stat._dates = [now, later]
        base_df = pd.DataFrame(data=base_stat.statistic, index=base_stat._dates, columns=base_stat._symbols)
        base_df_no_time = pd.DataFrame(data=base_stat.statistic, columns=base_stat._symbols)
        base_df_no_col = pd.DataFrame(data=base_stat.statistic, index=base_stat._dates, columns=[1,2,3,4])
        
        try:
            base_stat.check_data_validity(base_df) # check if this fails
        except:
            assert False, "Check with the correct data should not fail"

        try:
            base_stat.check_data_validity(base_df_no_col)
            assert False, "Value Error Expected"
        except ValueError as e:
            assert "must be string" in str(e)

        try:
            base_stat.check_data_validity(base_df_no_time)
            assert False, "Value Error Expected"
        except ValueError as e:
            assert "must be Time" in str(e)

        try:
            base_stat.check_data_validity(np.ones((2,3)))
            assert False, "Value Error Expected"
        except ValueError as e:
            assert "series or dataframe" in str(e)
