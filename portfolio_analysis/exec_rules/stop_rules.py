import pandas as pd

def static(bps: float, type: str):
    if type is None:
        raise ValueError("No trading type provided (must be +1 for 'long' or -1 for 'short')")
    def stop(price: pd.Series) -> pd.Series:
        return price * (1 - type * bps / 10000)
    return stop

def dynamic(bps: float, window: int, type: str):
    if type is None:
        raise ValueError("No trading type provided (must be +1 for 'long' or -1 for 'short')")
    def stop(price: pd.Series) -> pd.Series:
        rolling_price = price.rolling(window).max()
        if type == 1:
            rolling = rolling_price.max()
        if type == -1:
            rolling = rolling_price.min()
        return rolling * (1 - type * bps / 10000)
    return stop

def var_dynamic(bps: float, window: int, pct: float, type: str):
    if type is None:
        raise ValueError("No trading type provided (must be +1 for 'long' or -1 for 'short')")
    def stop(price: pd.Series) -> pd.Series:
        rolling_price = price.rolling(window)
        ref_price = rolling_price[0]
        if type == 'long':
            rolling = rolling_price.max()
        if type == 'short':
            rolling = rolling_price.min()
        return rolling * (1 - type * bps / 10000) + pct * (rolling - ref_price)
    return stop


STOP_RULES = {'static': static,
              'dynamic': dynamic,
              'var_dynamic': var_dynamic}