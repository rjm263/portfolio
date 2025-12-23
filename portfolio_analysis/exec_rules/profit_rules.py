import pandas as pd

def static(bps: float, type: str):
    if type is None:
        raise ValueError("No trading type provided (must be +1 for 'long' or -1 for 'short')")
    def profit(price: pd.Series) -> pd.Series:
        return price * (1 + type * bps / 10000)
    return profit


PROFIT_RULES = {'static': static}