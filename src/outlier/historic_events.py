from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List, Set

import base_outlier_set
import bs4 as bs
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyperclip as pc
import requests
from schema import Optional as schopt
from schema import Schema


class HistoricEvent:
    def __init__(
        self,
        date: pd.Timestamp,
        title: str = "",
        note: str = "",
        link: str = "",
        symbols: Set[str] = [],
        deleted: bool = False,
    ):
        """Initializes historic event with the most importnat characteristics

        Args:
            date (pd.Timestamp): date of occurence
            title (str): Title of the event
            note (str): Notes of the event
            link (str): Link to the website where the information was found
        """
        self._date = date.floor("d")
        self._title = title
        self._note = note
        self._link = link
        self._symbols: Set[str] = set(symbols)
        self._deleted: bool = deleted

    def __str__(self):
        note = self._note
        if len(note) >= 50:
            note = self._note[:47] + "..."
        return f"{self.title}, {note}"

    def to_dict(self) -> Dict[str, any]:
        dict_ = {}
        dict_["date"] = self.date.strftime("%Y-%m-%d")
        dict_["title"] = self.title
        dict_["note"] = self.note
        dict_["link"] = self.link
        dict_["symbols"] = list(self.symbols)
        dict_["deleted"] = self.deleted
        return dict_

    @staticmethod
    def get_schema():
        # check against schema
        schema = Schema(
            {
                "date": str,
                "title": str,
                "note": str,
                "link": str,
                schopt("symbols"): [str],
                schopt("deleted"): bool,
            }
        )

        return schema

    def __eq__(self, other: HistoricEvent):
        """neglect real and note properties in comparison"""
        return self.date == other.date and self.title.lower() == other.title.lower()

    def __lt__(self, other: HistoricEvent):
        return (self.date, self.title.lower()) < (other.date, other.title.lower())

    def __le__(self, other: HistoricEvent):
        return (self.date, self.title.lower()) <= (other.date, other.title.lower())

    def __gt__(self, other: HistoricEvent):
        return (self.date, self.title.lower()) > (other.date, other.title.lower())

    def __ge__(self, other: HistoricEvent):
        return (self.date, self.title.lower()) >= (other.date, other.title.lower())

    def __hash__(self) -> int:
        return tuple.__hash__((self.date, self.title.lower()))

    @property
    def date(self) -> pd.Timestamp:
        return self._date.floor("d")

    @date.setter
    def date(self, value: str):
        self._date = value

    @property
    def symbols(self) -> Set[str]:
        return self._symbols

    @symbols.setter
    def symbols(self, value: Iterable[str]):
        self._symbols = set(value)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value

    @property
    def note(self) -> str:
        return self._note

    @note.setter
    def note(self, value: str):
        self._note = value

    @property
    def link(self) -> str:
        return self._link

    @link.setter
    def link(self, value: str):
        self._link = value

    @property
    def deleted(self) -> bool:
        return self._deleted

    @deleted.setter
    def deleted(self, value: bool):
        self._deleted = value


class HistoricEventSet(base_outlier_set.BaseOutlierSet):
    """Outlier Detector which only loads the csv file of outliers"""

    def __init__(self, path: Path | str | None = None):
        base_outlier_set.BaseOutlierSet.__init__(self)
        if path is not None and Path(path).exists():
            self.load_events(path)
        self._name = "Historic Events"
        self._events: Dict[pd.Timestamp, List[HistoricEvent]] = {}
        self._text_element = None
        self._init_path = Path(path)
        self._cache = Path(self._init_path.parent)

    def __len__(self):
        if self._events is None:
            return 0
        return len(self._events)

    def add_event(self, event: HistoricEvent):
        if event.date in self._events.keys():
            self._events[event.date].append(event)
        else:
            self._events[event.date] = [event]

    def store_events(self, path: Path | str):
        """Stores set of outliers at the given path

        Args:
            path (Path | str): Path to store outliers at

        Raises:
            FileExistsError: Path to parent director must exists
        """

        path = Path(path)
        if not path.parent.exists():
            raise FileExistsError("The path to the parent must already exists.")

        lst = self._dict_to_list(self._events)

        with path.open("w") as target:
            json.dump(lst, target)

    def _dict_to_list(
        self, dic: Dict[pd.Timestamp, List[HistoricEvent]]
    ) -> List[Dict[str, any]]:
        lst = []
        for _, events in dic.items():
            for event in events:
                lst.append(event.to_dict())

        return lst

    def _list_to_dict_of_events(
        self, ls: List[Dict[str, any]]
    ) -> Dict[pd.Timestamp, List[HistoricEvent]]:
        schema = Schema([HistoricEvent.get_schema()])
        if not schema.validate(ls):
            raise RuntimeError("Error while loading events")

        dict: Dict[pd.Timestamp, List[HistoricEvent]] = {}
        for elem in ls:
            date = pd.Timestamp(elem["date"]).floor("d")
            elem["date"] = date
            if date in dict.keys():
                dict[date].append(HistoricEvent(**elem))
            else:
                dict[date] = [HistoricEvent(**elem)]

        return dict

    def load_events(self, path: Path | str):
        """Load events from backup file

        Args:
            path (Path | str): Path to look for the file

        Raises:
            FileExistsError: File at path must exist
            RuntimeError: File must have the right schema
        """

        path = Path(path)
        if not path.exists():
            raise FileExistsError(f"The path {str(path)} does not exist.")

        with path.open() as source:
            _list = json.load(source)

        self._events = self._list_to_dict_of_events(_list)

    def open_event_links(self, date: pd.Timestamp, symbol: str | None = None):
        """Opens a list of new browser windows and tabs that could be correlated to the event

        Args:
            date (pd.Timestamp): Date to investigate
            symbol (str | None, optional): Symbol to be investigated. Defaults to None.
        """
        wallstreet_journal = (
            f'https://www.wsj.com/news/archive/{date.strftime("%Y/%m/%d")}'
        )
        webbrowser.open_new_tab(wallstreet_journal)

        date_str = date.strftime("%Y-%m-%d")
        pc.copy(date_str)
        if symbol is not None:
            symbol_name = self.get_stock_name(symbol)
            google_search = f"https://www.google.com/search?client=firefox-b-d&q={date_str}+{symbol_name}"
            webbrowser.open_new_tab(google_search)
            pc.copy(f"{date_str} {symbol_name}")

    def cmd_add_symbol(self, date: pd.Timestamp, symbol: str):
        date = date.floor("d")
        if date in self._events:
            try:
                num = len(self._events[date])
                current_events = self._events[date]
                if current_events > 1:
                    for i, event in enumerate(self._events[date]):
                        click.echo("\nAvailable Events:")
                        click.echo(f"[{i}] : {str(event)}")

                    choices = [k for k in np.arange(num)]
                    idx = click.prompt(
                        "What to what event the symbol should be added?",
                        default=0,
                        type=click.Choice(choices=choices, case_sensitive=False),
                        prompt_suffix="?",
                        show_default=True,
                        show_choices=True,
                    )
                else:
                    idx = 0

                current_events[idx].symbols.add(symbol)
                click.echo(f"Added {symbol} to {str(current_events[idx])}")
            except click.Abort:
                click.echo("\nAbort editing events.")
                return
            except Exception as e:
                click.echo(
                    f"Error occurred during add event, previous state was restored. {e}"
                )
                return

    def get_stock_name(self, symbol: str):
        symbols_path = self._cache / "symbol_name_map.csv"
        if symbols_path.exists():
            symbol_df = pd.read_csv(symbols_path)
        else:
            # Download stocks names from S&P500 page on wikipedia
            resp = requests.get(
                "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            soup = bs.BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"class": "wikitable sortable"})
            symbols = []
            names = []
            for row in table.findAll("tr")[1:]:
                sym = row.findAll("td")[0].text
                name = row.findAll("td")[1].text
                symbols.append(sym)
                names.append(name)
            symbols = [s.replace("\n", "") for s in symbols]
            symbols = symbols + ["SPY"]
            names = names + ["S&P 500 Index"]

            # Fix wrong names (BF.B, BRK.B)
            symbols = [s.replace(".", "-") for s in symbols]
            symbol_df = pd.DataFrame.from_dict({"symbols": symbols, "names": names})
            symbol_df.to_csv(symbols_path, index=False)

        symbol_df = symbol_df.set_index("symbols")
        return symbol_df.loc[symbol, "names"]

    def cmd_edit_event(self, date: pd.Timestamp, symbol: str | None = None):
        date = date.floor("d")
        default_note = ""
        default_link = ""
        default_title = None
        if date in self._events:
            try:
                num = len(self._events[date])
                for i, event in enumerate(self._events[date]):
                    click.echo(f"[{i}] : {str(event)}")
                choices = [f"{k}" for k in np.arange(num)] + ["new"]
                idx = click.prompt(
                    "What event should be changed ",
                    default="0",
                    type=click.Choice(choices=choices, case_sensitive=False),
                    prompt_suffix="?",
                    show_default=True,
                    show_choices=True,
                )

            except click.Abort:
                click.echo("\nAbort editing events.")
                return
            except Exception as e:
                click.echo(
                    f"Error occurred during change event, previous state was restored. {e}"
                )
                return

            if idx == "new":
                event = HistoricEvent(date)
                event_list = self._events[date].copy() + [event]
            else:
                event = self._events[date][int(idx)]
                event_list = self._events[date]
                default_note = event.note
                default_link = event.link
                default_title = event.title

        else:
            # no events for that day exist
            event = HistoricEvent(date)
            event_list = [event]

        def non_empty(in_: str) -> str:
            in_ = in_.strip()
            if len(in_) == 0:
                click.echo("Empty strings are not allowed")
                raise click.BadParameter("Empty arguments are not allowed")
            return in_

        while True:
            try:
                title = click.prompt(
                    "Enter Title", value_proc=non_empty, default=default_title
                )
            except click.BadParameter:
                continue
            except click.Abort:
                click.echo("\nAbort editing events.")
                return
            break

        try:
            note = click.prompt("Enter Note", default=default_note)
        except click.Abort:
            print()
            note = ""

        try:
            link = click.prompt("Paste Link", default=default_link)
        except click.Abort:
            print()
            link = ""

        try:
            click.echo("\n\n---- Summary ----")
            click.echo(f"Title: {title}")
            click.echo(f"Note: {note}")
            click.echo(f"Link: {link}")
            if click.confirm("Are the above declarations correct?", default=True):
                event.title = title
                event.note = note
                event.link = link
                if symbol is not None:
                    event.symbols.add(
                        symbol
                    )  # as this is a set no duplications will be made
                self._events[date] = (
                    event_list  # add event list to make sure changed element is not added
                )

        except click.Abort:
            click.echo("\nAbort editing events.")
            return

    def draw_events(self, ax: plt.Axes, date: pd.Timestamp):
        """Draws an event text box for a given date at the top of the axis

        Args:
            ax (plt.Axes): Axis to plot onto
            date (pd.Timestamp): Date to look for events
        """

        if date.floor("d") in self._events.keys():
            text = ""
            for event in self._events[date.floor("d")]:
                text = str(event) + "\n"
            self._text_element = ax.text(
                x=0.98,
                y=0.98,
                s=text,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize="large",
            )
        else:
            if self._text_element is not None:
                try:
                    self._text_element.remove()
                except:  # noqa E722
                    pass  # ingnore failor here as it's not important
