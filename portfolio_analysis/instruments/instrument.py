#%%
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import yfinance as yf


def price_history(symbol: str, start: datetime, end: datetime=datetime.today()):
    """
    Return historic asset price as a dataframe.

    Parameters:
    ----------
    symbol: str
        The symbol of the financial asset.
    start: datetime
        The start date. Format: 2000-10-01
    end: datetime
        The end date; default is today. Format: 2000-10-01
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)['Close']
    df.columns.name = None
    df.index.name = 'Date'
    return df

class Instrument(ABC):
    """
    Abstract base class for financial instruments.

    Parameters:
    ----------
    name: str
        Name of the asset. Can be any, no need to coincide with symbol.
    symbol: str
        The symbol of the financial asset.
    amount: float
        The amount of shares for the financial asset.
    timestamp: str
        Time the asset was added to the portfolio. Format: 2000-10-01
    notes: str
        Own notes about the financial asset.
    """

    category: str = None

    def __init__(self, name: str, symbol: str, timestamp: str=None, notes: str=None):
        self.name = name
        self.symbol = symbol
        self.timestamp = datetime.today() if timestamp is None else datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        self.notes = notes if notes is not None else ''

        from ..portfolio.portfolio import Portfolio
        active_portfolio = Portfolio.get_active()
        if active_portfolio is not None:
            active_portfolio.add_instrument(self)
        else: print(f'No active portfolio set! {self.name} not registered.')

    @classmethod
    def add_multiple(cls, list_args):
        """Add multiple assets of the same type at once."""
        instances = []
        for args in list_args:
            instances.append(cls(**args))
        return instances

    @abstractmethod
    def get_value(self) -> pd.DataFrame:
        """Return historic value of asset as dataframe."""
        pass

    @abstractmethod
    def get_returns(self) -> pd.DataFrame:
        """Return historic returns of asset as dataframe."""
        pass

    @abstractmethod
    def get_info(self):
        """Return asset info as per Yahoo Finance."""
        pass

    def get_notes(self) -> str:
        """Return own notes attached to asset."""
        return f"Notes: {self.notes}\nDate added: {self.timestamp.strftime('%Y-%m-%d')}"
