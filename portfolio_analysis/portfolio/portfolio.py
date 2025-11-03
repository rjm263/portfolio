import pandas as pd


class Portfolio:
    """
    Class for the creation of a portfolio.

    A Portfolio instance p can be set as 'active' by calling set_active(p). All assets
    subsequently instantiated are registered to the active portfolio.
    """
    active_portfolio = None

    def __init__(self):
        self.stocks = {}
        self.savings_plans = {}
        self.bonds = {}
        self.futures = {}
        self.options = {}
        self.cash = pd.DataFrame({'Origin': [], 'Amount': []})
        self.cash.index.name = 'Date'

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
        from ..instruments.stocks import Stock
        from ..instruments.savings_plans import SavingsPlan
        from ..instruments.bonds import Bond
        from ..instruments.futures import Future
        from ..instruments.options import Option
        if isinstance(instrument, Stock):
            self.stocks[instrument.name] = instrument
        elif isinstance(instrument, SavingsPlan):
            self.savings_plans[instrument.name] = instrument
        elif isinstance(instrument, Bond):
            self.bonds[instrument.name] = instrument
        elif isinstance(instrument, Future):
            self.futures[instrument.name] = instrument
        elif isinstance(instrument, Option):
            self.options[instrument.name] = instrument
        else: raise ValueError('Instrument type not included in portfolio!')

    def add_cash(self, origin, amount):
        """Add amount of cash from origin to cash account."""
        today = pd.to_datetime("today")
        self.cash.loc[today] = [origin, amount]

    def get_total_value(self):
        """Return total value per asset class and of portfolio."""
        total_stocks = sum([value.get_value().iloc[-1].item() for _, value in self.stocks.items()])
        total_savings_plans = sum([value.get_value().iloc[-1].item() for _, value in self.savings_plans.items()])
        total_bonds = sum([value.get_value().iloc[-1].item() for _, value in self.bonds.items()])
        total_futures = sum([value.get_value().iloc[-1].item() for _, value in self.futures.items()])
        total_options = sum([value.get_value().iloc[-1].item() for _, value in self.options.items()])
        total_value = (total_stocks + total_savings_plans + total_bonds + total_futures +
                       total_options + self.cash['Amount'].sum())
        print(f'Value stocks: {total_stocks:.2f}\n'
              f'Value savings plans: {total_savings_plans:.2f}\n'
              f'Value bonds: {total_bonds:.2f}\n'
              f'Value futures: {total_futures:.2f}\n'
              f'Value options: {total_options:.2f}\n\n'
              f'Total value: {total_value:.2f}'
              )

