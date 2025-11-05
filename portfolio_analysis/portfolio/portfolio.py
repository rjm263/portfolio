import pandas as pd
from datetime import datetime


class Portfolio:
    """
    Class for the creation of a portfolio.

    A Portfolio instance p can be set as 'active' by calling set_active(p). All assets
    subsequently instantiated are registered to the active portfolio.
    """
    active_portfolio = None

    def __init__(self):
        self.stocks = []
        self.savings_plans = []
        self.bonds = []
        self.futures = []
        self.options = []
        self.cash = pd.DataFrame({'Origin': [], 'Amount': []})
        self.cash.index.name = 'Date'
        self.earliest_date = datetime.today()
        self.instruments = {'Stocks': self.stocks, 'SavingsPlans': self.savings_plans,
                            'Bonds': self.bonds, 'Futures': self.futures, 'Options': self.options}

    @classmethod
    def set_active(cls, portfolio):
        """Set portfolio into which instruments are registered."""
        cls.active_portfolio = portfolio

    @classmethod
    def get_active(cls):
        """Return active portfolio."""
        return cls.active_portfolio

    def add_instrument(self, instrument):
        """Add instruments to respective list."""
        category = getattr(instrument, 'category', None)
        if category and hasattr(self, category):
            getattr(self, category).append(instrument)
            if instrument.timestamp < self.earliest_date:
                self.earliest_date = instrument.timestamp
        else: raise ValueError(f'Instrument {type(instrument).__name__} has unknown/missing category!')

    def add_cash(self, origin, amount):
        """Add amount of cash from origin to cash account."""
        today = pd.to_datetime("today")
        self.cash.loc[today] = [origin, amount]

    def get_current_value(self) -> float:
        """Return total value of portfolio as of now."""
        total_instruments = sum([sum(_current_value(s)) for _,s in self.instruments.items()])
        total_cash = self.cash['Amount'].sum()
        return (total_instruments + total_cash).item()

    def get_current_returns(self) -> float:
        """Return total returns of portfolio as of now."""
        total_portfolio_value = self.get_current_value()
        return sum([sum([a * b for a, b in zip(_current_returns(s), _current_weights(s, total_portfolio_value))
                         ]) for _,s in self.instruments.items()])

    def get_history(self, start=None, end=None) -> pd.DataFrame:
        """Return price history for all assets, grouped by asset class."""
        if not start: start = self.earliest_date.date()
        if not end: end = datetime.today().date()
        dfs = {}
        for key, a in self.instruments.items():
            if not a: continue
            frms = [x for s in a for x in [s.get_value().loc[start:end], s.get_returns().loc[start:end]]]
            df = pd.concat(frms, axis=1)
            dfs[key] = df
        if not dfs: raise ValueError('The portfolio is empty!')
        return pd.concat(dfs, axis=1)

    def get_info(self) -> pd.DataFrame:
        """Return list of all assets and their variables."""
        cols = ['name', 'symbol', 'price', 'amount']
        dfs = {}
        for key, a in self.instruments.items():
            if key != 'Bonds':
                d1 = {c: [getattr(s, c, None) for s in a] for c in cols}
            else:
                cols_b = ['name', 'symbol', 'price', 'face_value', 'amount']
                d1 = {c: [getattr(s, c, None) for s in a] for c in cols_b}
            d2 = {'current_value': [s.get_value().iloc[-1].item() for s in a],
                  'current_returns': [s.get_returns().iloc[-1].item() for s in a]}
            df = pd.DataFrame(d1 | d2)
            df.set_index('name', inplace=True)
            dfs[key] = df
        return pd.concat(dfs, axis=0)



############################################################################################################
def _current_value(l):
    current_value = []
    for el in l:
        df = el.get_value()
        if not df.empty:
            value = df.iloc[-1].item()
        else: value = 0
        current_value.append(value)
    return current_value


def _current_returns(l):
    current_returns = []
    for el in l:
        df = el.get_returns()
        if not df.empty:
            returns = df.iloc[-1].item()
        else: returns = 0
        current_returns.append(returns)
    return current_returns

def _current_weights(l, total_portfolio_value):
    return [v / total_portfolio_value for v in _current_value(l)]





