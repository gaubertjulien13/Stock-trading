"""Tune the only setup that showed a real edge, with a strict train/holdout split.

Parameters are chosen on 2008-2017 only. 2018-2026 is touched once, at the end, to
see whether the choice survives. Anything that only works in-sample is discarded.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build
from portfolio_test import equity_curve, stats

pd.set_option('display.width', 250)

TRAIN = (pd.Timestamp('2008-01-01'), pd.Timestamp('2017-12-31'))
TEST = (pd.Timestamp('2018-01-01'), pd.Timestamp('2026-07-31'))


def momentum_mask(f, base, close, lookback=252, pct=0.80, skip=21,
                  require_above200=True, require_rising200=False,
                  max_vol=None, regime=False, d=None):
    # Standard academic momentum skips the most recent month to avoid short-term reversal
    mom = (close.shift(skip) / close.shift(skip + lookback) - 1.0)
    mom = mom.where(base)
    rank = mom.rank(axis=1, pct=True)
    m = base & (rank >= pct)
    if require_above200:
        m &= f['above_200']
    if require_rising200:
        m &= f['sma200_slope'] > 0
    if max_vol is not None:
        m &= f['vol_ann'] <= max_vol
    if regime:
        m &= f['spy_above_200'].fillna(False)
    return m


def run(mask, d, hold, max_pos, window):
    port, w = equity_curve(mask, d, hold_days=hold, max_pos=max_pos)
    port = port.loc[window[0]:window[1]]
    spy = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0).loc[window[0]:window[1]]
    return stats(port, spy, '', (w.loc[window[0]:window[1]] > 0).sum(axis=1).mean()), port


def sweep(d, f, base, close, grid, window, label):
    rows = []
    for name, kw, hold, max_pos in grid:
        m = momentum_mask(f, base, close, d=d, **kw)
        if m.values.sum() < 200:
            continue
        s, _ = run(m, d, hold, max_pos, window)
        s['strategy'] = name
        s['hold_d'] = hold
        s['max_pos'] = max_pos
        rows.append(s)
    df = pd.DataFrame(rows)
    cols = ['strategy', 'hold_d', 'max_pos', 'CAGR_%', 'SPY_CAGR_%', 'excess_%',
            'vol_%', 'sharpe', 'maxDD_%', 'avg_positions']
    print()
    print('=' * 150)
    print(label)
    print('=' * 150)
    print(df[cols].sort_values('excess_%', ascending=False).round(2).to_string(index=False))
    return df


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    base = build(d, f)

    grid = []
    for lb in (126, 252):
        for pct in (0.70, 0.80, 0.90):
            for hold in (63, 126, 252):
                grid.append((f'mom{lb} top{int((1-pct)*100)}%', dict(lookback=lb, pct=pct),
                             hold, 20))
    sweep(d, f, base, close, grid, TRAIN, 'TRAIN 2008-2017  --  lookback / selectivity / hold')

    grid2 = [
        ('base: mom252 top20 above200', dict(), 126, 20),
        ('+ rising 200dma', dict(require_rising200=True), 126, 20),
        ('+ vol cap 60%', dict(max_vol=60), 126, 20),
        ('+ SPY regime filter', dict(regime=True), 126, 20),
        ('+ rising200 + vol cap', dict(require_rising200=True, max_vol=60), 126, 20),
        ('+ all three', dict(require_rising200=True, max_vol=60, regime=True), 126, 20),
        ('no above200 filter', dict(require_above200=False), 126, 20),
        ('no skip month', dict(skip=0), 126, 20),
        ('10 positions', dict(), 126, 10),
        ('30 positions', dict(), 126, 30),
        ('40 positions', dict(), 126, 40),
    ]
    sweep(d, f, base, close, grid2, TRAIN, 'TRAIN 2008-2017  --  filters and concentration')

    # ------------------------------------------------------------------- holdout
    print()
    print('=' * 150)
    print('HOLDOUT 2018-2026  --  the configurations above, evaluated once')
    print('=' * 150)
    finalists = [
        ('mom252 top20%, 20 pos, 6m hold', dict(), 126, 20),
        ('mom252 top20% + regime filter', dict(regime=True), 126, 20),
        ('mom252 top20% + vol cap', dict(max_vol=60), 126, 20),
        ('mom252 top10%, 20 pos', dict(pct=0.90), 126, 20),
        ('mom126 top20%, 3m hold', dict(lookback=126), 63, 20),
        ('mom252 top20%, 30 pos', dict(), 126, 30),
    ]
    rows = []
    for name, kw, hold, mp in finalists:
        m = momentum_mask(f, base, close, d=d, **kw)
        s_tr, _ = run(m, d, hold, mp, TRAIN)
        s_te, port = run(m, d, hold, mp, TEST)
        rows.append({
            'strategy': name,
            'train_CAGR': s_tr['CAGR_%'], 'train_excess': s_tr['excess_%'],
            'train_sharpe': s_tr['sharpe'],
            'test_CAGR': s_te['CAGR_%'], 'test_SPY': s_te['SPY_CAGR_%'],
            'test_excess': s_te['excess_%'], 'test_sharpe': s_te['sharpe'],
            'test_maxDD': s_te['maxDD_%'], 'SPY_maxDD': s_te['SPY_maxDD_%'],
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ------------------------------------------------- year by year for the finalist
    m = momentum_mask(f, base, close, d=d)
    _, port = run(m, d, 126, 20, (pd.Timestamp('2008-01-01'), TEST[1]))
    spy = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0).reindex(port.index)
    yr = pd.DataFrame({'strategy': port, 'spy': spy}).groupby(port.index.year).apply(
        lambda g: pd.Series({'strategy_%': ((1 + g['strategy']).prod() - 1) * 100,
                             'spy_%': ((1 + g['spy']).prod() - 1) * 100}))
    yr['excess_%'] = yr['strategy_%'] - yr['spy_%']
    print()
    print('=' * 150)
    print('YEAR BY YEAR  --  mom252 top20%, 20 positions, 6-month holds')
    print('=' * 150)
    print(yr.round(1).to_string())
    print()
    print(f'Years beating SPY: {(yr["excess_%"] > 0).sum()} / {len(yr)}')
    print(f'Median annual excess: {yr["excess_%"].median():.1f}%')


if __name__ == '__main__':
    main()
