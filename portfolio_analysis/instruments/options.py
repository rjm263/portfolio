from .instrument import Instrument, price_history
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# add value+return for 'write' or create new class (?)

class Option(Instrument):
    """
    Class for options; inherits from Instrument class.

    Parameters:
    ----------
    name: str
        Name of the asset. Can be any, no need to coincide with symbol.
    symbol: str
        The symbol of the financial asset.
    strike: float
        The strike price of the options contract.
    maturity: str
        The maturity of the options contract. Format: 2000-10-01
    premium: float
        The premium that has to be paid. (Support for writing to come...)
    amount: float
        The amount of identical options contracts purchased on the underlying.
    quantity: int
        The amount of shares for one options contract (default are 100).
    position: str
        Call position ('c') or put position ('p').
    timestamp: str
        Time the asset was added to the portfolio. Format: 2000-10-01
    notes: str
        Own notes about the financial asset.
    """
    def __init__(self, name, symbol, strike: float, maturity: str, premium: float, amount=1.0,
                 quantity: int=100, position: str='c', timestamp=None, notes=None):
        super().__init__(self, name, symbol, amount, timestamp, notes)
        self.strike = strike
        self.maturity = datetime.strptime(maturity, '%Y-%m-%d')
        self.premium = premium
        self.quantity = quantity
        if position not in ('c','p'): raise ValueError('Position argument must be \'c\' or \'p\'!')
        self.position = position

    def get_value(self):
        prices = price_history(self.symbol, self.timestamp)
        today = datetime.today().date()
        maturity_today = pd.date_range(start=self.maturity, end=today, freq='1D')
        if self.position == 'c':
            value_df = np.max(0, (prices - self.strike - self.premium) * self.amount * self.quantity)
        else:
            value_df = np.max(0, (self.strike - prices - self.premium) * self.amount * self.quantity)
        if self.maturity < today:
            value_df.loc[maturity_today] = value_df.loc[self.maturity].item()
        return value_df

    def get_returns(self, log=False):
        returns = self.get_value() / (self.premium * self.amount * self.quantity)
        return returns if not log else np.log(1 + returns)

    def get_info(self):
        return yf.Ticker(self.symbol).info



