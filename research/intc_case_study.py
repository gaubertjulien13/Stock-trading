"""Reconstruct the INTC trade as it looked in real time.

Goal: separate the parts of the thesis that were knowable in advance from the
parts that are hindsight. Prints the drawdown path, the semis-industry backdrop,
and what a "beaten down + booming industry" screen would have flagged on each
date -- including all the dates where it would have been catastrophically early.
"""

import pandas as pd
import numpy as np
import yfinance as yf

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 300)

START = '2018-01-01'
END = '2026-08-01'

TICKERS = ['INTC', 'SMH', 'SPY', 'NVDA', 'AMD']


def load():
    df = yf.download(TICKERS, start=START, end=END, interval='1d',
                     progress=False, auto_adjust=True)
    close = df['Close'].dropna(how='all')
    return close


def drawdown_path(close):
    intc = close['INTC'].dropna()
    # Rolling 3-year peak = "the price used to be much higher" anchor
    peak_3y = intc.rolling(756, min_periods=200).max()
    dd = intc / peak_3y - 1.0
    return intc, peak_3y, dd


def main():
    close = load()
    intc, peak_3y, dd = drawdown_path(close)

    print('=' * 100)
    print('INTC PRICE PATH -- month-end snapshots')
    print('=' * 100)

    smh = close['SMH'].reindex(intc.index).ffill()
    spy = close['SPY'].reindex(intc.index).ffill()

    tbl = pd.DataFrame({
        'INTC': intc,
        'dd_from_3y_peak_%': dd * 100,
        'INTC_200dma': intc.rolling(200).mean(),
        'INTC_vs_200dma_%': (intc / intc.rolling(200).mean() - 1) * 100,
        # industry backdrop
        'SMH_12m_%': (smh / smh.shift(252) - 1) * 100,
        'SMH_above_200dma': smh > smh.rolling(200).mean(),
        # the divergence: stock lagging its own industry
        'INTC_minus_SMH_12m_%': ((intc / intc.shift(252) - 1) - (smh / smh.shift(252) - 1)) * 100,
        'INTC_6m_%': (intc / intc.shift(126) - 1) * 100,
    })

    monthly = tbl.resample('ME').last()
    print(monthly.loc['2021-01-01':].round(1).to_string())

    print()
    print('=' * 100)
    print('THE FALLING KNIFE PROBLEM')
    print('=' * 100)
    print('If the screen were: "INTC down >40% from 3y peak AND semis industry booming",')
    print('what would the forward return have been from each trigger date?')
    print()

    # Naive screen: deep drawdown + industry in uptrend. No timing filter.
    naive = (dd < -0.40) & tbl['SMH_above_200dma'].fillna(False)
    trig = naive[naive].index

    rows = []
    for horizon in (21, 63, 126, 252):
        fwd = intc.shift(-horizon) / intc - 1.0
        sel = fwd.reindex(trig).dropna()
        if len(sel):
            rows.append({
                'horizon_days': horizon,
                'n': len(sel),
                'mean_%': sel.mean() * 100,
                'median_%': sel.median() * 100,
                'win_rate_%': (sel > 0).mean() * 100,
                'worst_%': sel.min() * 100,
                'best_%': sel.max() * 100,
            })
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print()
    print('First trigger date of naive screen:', trig[0].date() if len(trig) else 'none',
          '| price', round(float(intc.loc[trig[0]]), 2) if len(trig) else '')
    print('Ultimate low:', intc.loc['2022-01-01':].idxmin().date(),
          '| price', round(float(intc.loc['2022-01-01':].min()), 2))
    print('Current:', intc.index[-1].date(), '| price', round(float(intc.iloc[-1]), 2))

    if len(trig):
        first = trig[0]
        low = intc.loc['2022-01-01':].min()
        print(f'\nDrawdown suffered by a buyer at first trigger before the bottom: '
              f'{(low / float(intc.loc[first]) - 1) * 100:.1f}%')

    print()
    print('=' * 100)
    print('WHAT A STABILIZATION FILTER WOULD HAVE DONE')
    print('=' * 100)
    print('Add: price must reclaim its 200dma (evidence the decline stopped).')
    print()

    dma200 = intc.rolling(200).mean()
    above = intc > dma200
    reclaim = above & (~above.shift(1).fillna(False))
    timed = naive.shift(1).fillna(False) | naive  # deep dd context recently true
    # require the deep-drawdown context to have been true within past 6 months
    dd_ctx = (dd < -0.40).rolling(126, min_periods=1).max().astype(bool)
    timed_trig = reclaim & dd_ctx & tbl['SMH_above_200dma'].fillna(False)

    tt = timed_trig[timed_trig].index
    print('Trigger dates:', [str(d.date()) for d in tt])
    rows = []
    for horizon in (21, 63, 126, 252):
        fwd = intc.shift(-horizon) / intc - 1.0
        sel = fwd.reindex(tt).dropna()
        if len(sel):
            rows.append({
                'horizon_days': horizon, 'n': len(sel),
                'mean_%': sel.mean() * 100, 'median_%': sel.median() * 100,
                'win_rate_%': (sel > 0).mean() * 100,
                'worst_%': sel.min() * 100, 'best_%': sel.max() * 100,
            })
    if rows:
        print(pd.DataFrame(rows).round(1).to_string(index=False))

    # Max adverse excursion for the timed entries
    print()
    for d in tt:
        entry = float(intc.loc[d])
        fwd_path = intc.loc[d:].iloc[:253]
        mae = (fwd_path.min() / entry - 1) * 100
        mfe = (fwd_path.max() / entry - 1) * 100
        print(f'  {d.date()} entry {entry:7.2f} | worst drawdown next 12m {mae:6.1f}% | best {mfe:7.1f}%')


if __name__ == '__main__':
    main()
