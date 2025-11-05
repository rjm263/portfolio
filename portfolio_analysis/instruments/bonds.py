from .instrument import Instrument
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import yfinance as yf

class Bond(Instrument):
    """
    Class for bonds; inherits from Instrument class.

    Parameters:
    ----------
    name: str
        Name of the asset. Can be any, no need to coincide with symbol.
    maturity: str
        The date of maturity. Format: 2000-10-01
    face_value: float
        The face value of the bond.
    coupon: float
        The coupon of the bond as a percentage on face value.
    price: float
        The price of the asset at time of instantiation.
    start_date: str
        Date of purchase of the bond. Format: 2000-10-01
    frequency: int
        Frequency of interest payments in months.
    symbol: str
        The symbol of the bond, in case it is listed.
    timestamp: str
        Time the asset was added to the portfolio. Format: 2000-10-01
    notes: str
        Own notes about the financial asset.
    """

    category = 'bonds'

    def __init__(self, name, maturity: str, face_value: float, coupon: float, amount: float=1.0, price: float=None, start_date: str=None,
                 frequency: int=None, symbol=None, timestamp=None, notes=None):
        super().__init__(name, symbol, timestamp, notes)
        self.start_date = datetime.today().date() if start_date is None else datetime.strptime(start_date, '%Y-%m-%d')
        self.maturity = datetime.strptime(maturity, '%Y-%m-%d')
        self.face_value = face_value
        self.amount = amount
        if price is None and coupon == 0: raise ValueError('Please provide a price for zero-coupon bonds!')
        self.price = face_value if price is None else price
        self.coupon = coupon
        self.frequency = 6 if frequency is None else frequency

    def get_value(self):
        today = datetime.today()
        if self.maturity <= today: today = self.maturity            # TD: realise money and delete bond instance
        dates = pd.date_range(start=self.start_date, end=today, freq='D')
        if self.coupon == 0:
            df = pd.DataFrame({self.name: [self.face_value * self.amount]*len(dates)}, index=pd.Index(dates, name='Date'))
            df.rename(columns={self.name: 'Value'}, inplace=True)
            df.columns = pd.MultiIndex.from_product([[self.name], df.columns])
            return df

        delta = relativedelta(today, self.start_date)
        n_periods = (delta.years * 12 + delta.months) // self.frequency
        period_coupon = self.face_value * self.coupon * (self.frequency / 12)
        value_periods = [(self.face_value + i * period_coupon) * self.amount for i in range(n_periods+1)]
        value_series = pd.Series(index=dates, dtype=float)
        for i in range(n_periods+1):
            start = self.start_date + pd.DateOffset(months=i * self.frequency)
            end = start + pd.DateOffset(months=self.frequency) - pd.Timedelta(days=1)
            if end > today: end = today
            value_series.loc[start:end] = value_periods[i]
        # For now, after maturity continue listing the final value
        value_series.loc[today:datetime.today()] = value_periods[n_periods]
        df = pd.DataFrame({self.name: value_series})
        df.index.name = 'Date'
        df.rename(columns={self.name: 'Value'}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[self.name], df.columns])
        return df

    def get_returns(self, log=False):
        if self.coupon == 0:
            df = (self.get_value() / self.amount - self.price) / self.price
        else:
            df = (self.get_value() / self.amount - self.face_value) / self.face_value
        df.rename(columns={'Value': 'Returns'}, inplace=True)
        return df if not log else np.log(1 + df)

    def get_info(self):
        return yf.Ticker(self.symbol).info if self.symbol is not None else 'No info available.'




