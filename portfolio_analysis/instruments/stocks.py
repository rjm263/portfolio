from .instrument import Instrument, price_history
import numpy as np
import yfinance as yf

class Stock(Instrument):
    """
    Class for stocks; inherits from Instrument class.

    Parameters:
    ----------
    name: str
        Name of the asset. Can be any, no need to coincide with symbol.
    symbol: str
        The symbol of the financial asset.
    price: float
        The price of the asset at time of instantiation.
    amount: float
        The amount of shares for the asset at time of instantiation.
    timestamp: str
        Time the asset was added to the portfolio. Format: 2000-10-01
    notes: str
        Own notes about the financial asset.
    """
    def __init__(self, name, symbol, price: float, amount=1.0, timestamp=None, notes=None):
        super().__init__(name, symbol, amount, timestamp, notes)
        self.price = price

    def get_value(self):
        return price_history(self.symbol, self.timestamp) * self.amount

    def get_returns(self, log=False):
        returns = (price_history(self.symbol, self.timestamp) - self.price) / self.price
        return returns if not log else np.log(1 + returns)

    def get_info(self):
        return yf.Ticker(self.symbol).info