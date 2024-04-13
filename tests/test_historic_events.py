import pandas as pd
from historic_events import HistoricEvent, HistoricEventSet

class TestHistoricEventSet:
        
    def test_store_and_load_outliers(self):

        now = pd.Timestamp.now().floor('d')
        before = pd.Timestamp('1999-12-18 21:00:00').floor('d')
        recently = pd.Timestamp('2023-12-18 21:00:00').floor('d')
        e1 = HistoricEvent(now, "e1", "Test e1", "test")
        e2 = HistoricEvent(before, "e2", "Test e2", "link")
        e3 = HistoricEvent(recently, "e3", "Test e3", "link")

        # make sure the test runs smoothly
        path = "tests/testdata/cache/test_hist_event_cache.json"
        _set = HistoricEventSet(path)
        _set._events = {}
        _set.add_event(e1)
        _set.add_event(e2)
        _set.add_event(e3)
        _set.store_events(path)
        _set._events = {}
        _set.load_events(path)
        all = set()
        for _, e in _set._events.items():
            assert set(e) & {e1,e2,e3} != {}
            all = set(e).union(all)

        assert all == {e1, e2, e3}

    def test_open_event_links(self):
        he_set = HistoricEventSet()
        now = pd.Timestamp.now()
        he_set.open_event_links(now, "A")
    
    def test_edit_event(self):
        histset = HistoricEventSet()
        histset.cmd_edit_event(date=pd.Timestamp.now(), symbol="Test")
        for k, elem in histset._events.items():
            for ev in elem:
                print(ev)
        histset.cmd_edit_event(date=pd.Timestamp.now(), symbol="Test")
        for k, elem in histset._events.items():
            for ev in elem:
                print(ev)
        histset.cmd_edit_event(date=pd.Timestamp.now(), symbol="Test")
        for k, elem in histset._events.items():
            for ev in elem:
                print(ev)
    
    def test_get_snp500(self):
        
        # make sure the test runs smoothly
        path = "tests/testdata/cache/test_hist_event_cache.json"
        _set = HistoricEventSet(path)
        print(_set.get_stock_name("MMM"))

TestHistoricEventSet().test_get_snp500()

class TestHistoricEvent:
    
    def test_properties(self):


        date = pd.Timestamp('2017-01-01 12:00:00')
        title = "Test"
        note = "Note test"
        deleted = False
        event = HistoricEvent(date, "Test", "Note test", deleted=deleted)
        
        assert event.date == date.floor('d')
        assert event.note == note
        assert event.title == title
        assert event.deleted == deleted
        
        event.deleted = True
        assert event.deleted
        
        event.note = "Second"
        assert event.note == "Second"
        
        event.title = "A"
        assert event.title == "A"
    
    def test_dict_out(self):
        date = pd.Timestamp('2017-01-01 12:00:00')
        note = "Note test"
        outlier = HistoricEvent(date, "Test", "Note test")

        _dict = {'note' : note, 'date' : date.floor('d').strftime("%Y-%m-%d"), 'deleted' : False, 'symbols' : [] } 
        
        dict_out = outlier.to_dict()
        for k in _dict.keys():
            assert k in dict_out.keys()
            assert dict_out[k] == _dict[k]
        
    def test_comparison(self):

        smallest = pd.Timestamp('2017-01-01 09:00:00')
        small = pd.Timestamp('2017-01-01 12:00:00')
        middle = pd.Timestamp('2018-01-01 12:00:00')
        large = pd.Timestamp('2019-01-01 12:00:00')
        outlier_small = HistoricEvent(small, "Small Test", "Note test")
        outlier_small_2 = HistoricEvent(small, "small test", "Note test")
        outlier_small_3 = HistoricEvent(smallest, "small test", "Note test")
        outlier_smaller = HistoricEvent(small, "Amall Test", "Note test")
        outlier_middle = HistoricEvent(middle, "Midle Test", "Note test")
        outlier_large = HistoricEvent(large, "Large Test", "Note test")
        
        assert outlier_smaller < outlier_small
        assert outlier_smaller <= outlier_small
        assert outlier_small == outlier_small_3
        assert outlier_small == outlier_small_2
        assert outlier_small.__hash__() == outlier_small_2.__hash__()
        assert outlier_large > outlier_middle
        assert outlier_middle > outlier_small
        assert outlier_large >= outlier_middle
        assert outlier_middle >= outlier_small

        