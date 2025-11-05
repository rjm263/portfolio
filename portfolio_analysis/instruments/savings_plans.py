from .instrument import Instrument, price_history
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import yfinance as yf

class SavingsPlan(Instrument):
    """
    Class for savings plans; inherits from Instrument class.

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
    rate: float
        The savings rate in the currency of the underlying asset.
    next_date: str
        The next date of purchase from time of instantiation. Format: 2000-10-01
    frequency: int
        The frequency of execution in months.
    end_date: str
        The last date of execution. Format: 2000-10-01
    fees: float
        The fees for execution as percentage of savings rate.
    timestamp: str
        Time the asset was added to the portfolio. Format: 2000-10-01
    notes: str
        Own notes about the financial asset.
    """

    category = 'savings_plans'

    def __init__(self, name, symbol, price: float, amount, rate: float, next_date: str, frequency: int=1,
                 end_date: str=None, fees: float=None, timestamp=None, notes=None):
        super().__init__(name, symbol, timestamp, notes)
        self.price = price
        self.amount = amount
        self.rate = rate
        self.next_date = datetime.strptime(next_date, '%Y-%m-%d')
        self.frequency = relativedelta(months=frequency)
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.fees = 0 if fees is None else fees
        
    def get_value(self):
        accrued = []
        value_history = []
        dates = []
        prices = price_history(self.symbol, self.timestamp)
        today = datetime.today()
        current = self.timestamp
        nxt = self.next_date
        delta = relativedelta(days=4)   # markets are closed for at most 4 days in a row
        while current <= today - delta:
            if current is nxt:
                if current > today - delta and prices.loc[current:today].shape == (0,1):
                    continue
                else:
                    accrued.append(self.rate * (1 - self.fees) / price_history(self.symbol, current, current + delta).iat[0,0].item())
                nxt = nxt + self.frequency
            value_history.append((accrued[-1] + self.amount) * price_history(self.symbol, current - delta, current).iat[-1,0].item())
            dates.append(current.date())
            current = current + relativedelta(days=1)
        df = pd.DataFrame({self.symbol: value_history}, index=pd.to_datetime(dates))
        df.index.name = 'Date'
        df['Amount'] = [a + self.amount for a in accrued]
        df.rename(columns={self.symbol: 'Value'}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[self.symbol], df.columns])
        return df

    def get_returns(self, log=False):
        delta = relativedelta(datetime.today().date(), self.timestamp)
        periods = (delta.years * 12 + delta.months) // self.frequency.months
        df = (self.get_value() - (self.amount * self.price + periods * self.rate)) / (self.amount * self.price + periods * self.rate)
        df.rename(columns={'Value': 'Returns'}, inplace=True)
        return df if not log else np.log(1+df)

    def get_info(self):
        return yf.Ticker(self.symbol).info



        
