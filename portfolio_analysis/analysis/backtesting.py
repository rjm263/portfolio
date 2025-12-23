import pandas as pd
import numpy as np

from ..exec_rules.calendars import EVENT_CALENDARS
from ..exec_rules.date_rules import combine_rules, DATE_RULES
from ..exec_rules.stop_rules import STOP_RULES
from ..exec_rules.profit_rules import PROFIT_RULES
from ..exec_rules.volume_rules import VOLUME_RULES

def validate_signal(signal: pd.DataFrame):
    required = {'type', 'capital', 'entry_ts', 'profit_level', 'stop_loss', 'timeout',
                'datetime_rules', 'vol_rules', 'event_rules'}
    missing = required - set(signal.columns)
    if missing:
        raise ValueError(f'Missing columns {missing}')


def execute_trade(signal_row: pd.DataFrame, market_data: pd.DataFrame) -> dict: 
    """
    Single trade execution engine. Takes single row of DataFrame 'signal' and 
    DataFrame 'market_data' as input. These must contain the following cols:

    signal_row: (index = signal_id)
        - type: +1 for 'long' or -1 for 'short'
        - capital: amount invested in trade
        - entry_ts: pd.Timestamp
        - profit_level: dict {key: list} where key is 'static' and list contains args 
            as specified in profit_rules.py
        - stop_loss: dict {key: list} where key is 'static', 'dynamic' or 'var_dynamic'
            and list contains args as specified in stop_rules.py
        - timeout: pd.Timestamp; in days since entry
        - datetime_rules: rules that restrict trading to specific dates (e.g., only Mondays)
            default is empty list []
        - vol_rules: rules that restrict trading to specific volume bounds
            default is empty list []
        - event_rules: rules that exit trade before given even occurs (e.g., earnings)
            default is empty list []

    market_data: (index = dates)
        - OHLC with first letter capitalised (default in yfinance)

    Arguments:
        signal_row: pd.DataFrame
        market_data: pd.DataFrame
    """
    entry_ts = signal_row.entry_date
    if entry_ts not in market_data.index:
        raise ValueError('No price data available for entry date!')

    price_df = market_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    timestamps = price_df.index

    entry_iloc = timestamps.get_loc(entry_ts)
    entry_price = price_df['Close'].iloc[entry_iloc]

    timeout_ts = entry_ts + signal_row.timeout
    timeout_iloc = timestamps.searchsorted(timeout_ts, side='left')

    price_df = price_df.iloc[entry_iloc+1:timeout_iloc+1]
    if price_df.empty:
        raise ValueError('Timeout window is too short!')
    prices = price_df['Close'].to_numpy()

    k_p, v_p = signal_row.profit_level.items()
    profit_rules = PROFIT_RULES.get(k_p)(*(v_p + [signal_row.type]))
    profit_price = profit_rules(prices)
    
    k_s, v_s = signal_row.stop_loss.items()
    stop_rules = STOP_RULES.get(k_s)(*(v_s + [signal_row.type]))
    stop_price = stop_rules(prices)

    # For intraday trading, use OHLC logic
    highs = price_df['High'].to_numpy()
    lows = price_df['Low'].to_numpy()

    trade_type = signal_row.type
    if trade_type == 1:
        profit_hit = highs >= profit_price
        stop_hit = lows <= stop_price
    else:
        profit_hit = lows <= profit_price
        stop_hit = highs >= stop_price

    # Don't trade on market days/times ruled out by signal_row.datetime_rules
    date_rules = combine_rules(*[DATE_RULES.get(r) for r in signal_row.datetime_rules])
    date_mask = date_rules(price_df.index)

    # Only trade above/below volumes specified in signal_row.vol_rules
    vol_rules = combine_rules(*[VOLUME_RULES.get(r) for r in signal_row.vol_rules])
    vol_mask = vol_rules(price_df['Volume'])
    
    # If event in signal_row.event_calendar (e.g., earnings) occurs, exit before
    # NB: assuming that event occurs after market closures!
    event_datetimes = EVENT_CALENDARS.get(signal_row.event_rules)
    event_hit = price_df.index.isin(event_datetimes)

    exit_mask = stop_hit | profit_hit | event_hit
    exit_mask &= date_mask & vol_mask
    if exit_mask.any():
        exit_delta = exit_mask.argmax()
        exit_iloc = entry_iloc + exit_delta + 1

        if stop_hit[exit_delta]:
            exit_reason = 'stop'
        elif profit_hit[exit_delta]:
            exit_reason = 'profit'
        else:
            exit_reason = 'event'

    else:
        exit_reason = 'timeout'
        exit_iloc = timeout_iloc

    # Exit at bar close price (can be changed to touch price or next bar open price)
    exit_price = price_df['Close'].iloc[exit_iloc]

    return {'signal_id': signal_row.index,
            'type': signal_row.type,
            'capital': signal_row.capital,
            'entry_time': entry_ts,
            'exit_time': timestamps[exit_iloc],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason}


from multiprocessing import Pool, cpu_count
def run_backtest(signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Trade execution engine which parallelises into single trades executed by 
    execute_trade function. 

    Arguments:
        signals: pd.DataFrame
            structure must be as detailed in execute_trade docstring
        market_data: pd.DataFrame
            index must be datetime, must include a column 'Close' listing 
            closing prices
    """
    validate_signal(signals)
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            execute_trade,
            [(row, market_data) for row in signals.itertuples(index=False)]
        )
    return pd.DataFrame(results).set_index('signal_id')

def trades_ext(trades: pd.DataFrame, fee_model: function) -> pd.Series:
    """
    Amends output from run_backtest function by pnl, return and holding period per trade.

    Arguments:
        trades: pd.DataFrame
            output of run_backtest function
        fee_model: function
            function modelling fee structure of trades; must take invested capital as input
    """
    trades_ext = trades.copy()

    trades_ext['pnl'] = (trades['capital'] * trades['type'] * (trades['exit_price'] - trades['entry_price']) 
                     - fee_model(trades['capital']))
    trades_ext['return'] = trades['pnl'] / trades['capital']
    trades_ext['holding_period'] = (trades['exit_time'] - trades['entry_time']).dt.minutes

    out_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 'pnl', 'return', 'holding_period']
    return trades_ext[out_cols]


def trade_blotter(trades: pd.DataFrame) -> pd.Series:
    """
    Takes in a DataFrame 'trades' and extracts main metrics.

    Arguments:
        trades: pd.DataFrame
            must be output of trades_ext function
    """
    trades = trades.copy()
    
    trades['is_win'] = trades['pnl'] > 0

    stats_per_cat = trades.groupby('exit_reason')['return'].agg(['count', 'mean', 'std'])
    stats_per_type = trades.groupby('type')['return'].agg(['count', 'mean', 'std'])

    num_trades = len(trades)
    wins = trades['is_win'].sum()
    win_rate = wins / num_trades if num_trades > 0 else np.nan

    nom_profit = trades[trades['is_win'] == True]['pnl'].sum()
    nom_loss = trades[trades['is_win'] == False]['pnl'].sum()

    avg_loss = trades[trades['is_win'] == False]['return'].mean()

    expected_return = trades['return'].mean()
    expected_loss = -avg_loss

    volatility = trades['return'].std()

    avg_trading_days = trades['trading_days'].mean()
    
    return pd.Series(
        {'total_trades': num_trades,
         'win_rate': win_rate,
         'profit_exits': stats_per_cat.loc['profit', 'count'],
         'stop_exits': stats_per_cat.loc['stop', 'count'],
         'timeout_exits': stats_per_cat.loc['timeout', 'count'], 
         'pnl': nom_profit + nom_loss,
         'pnl_wins': nom_profit,
         'pnl_losses': -nom_loss,
         'expected_return': expected_return,
         'expected_loss': expected_loss,
         'avg_return_long': stats_per_type.loc[1, 'mean'],
         'avg_return_short': stats_per_type.loc[-1, 'mean'],
         'volatility': volatility,
         'avg_trading_days': avg_trading_days}
         )