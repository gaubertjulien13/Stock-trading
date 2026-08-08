"""Is there enough edge here to be worth trading? The decisive tests.

validate_assumptions.py showed that picking 20 of ~100 qualifying momentum names moves the
result by +/-3% CAGR depending purely on which arbitrary 20 you take. That is
selection noise, not edge. Two questions follow:

1. If the book is widened until selection noise disappears, what edge is left?
2. Is whatever remains persistent, or does it come from a handful of years?

Run:  venv/bin/python3 research/validate_edge.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build
from portfolio_test import stats
from tune_momentum import momentum_mask
from verify import delisting_returns, equity_curve

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 60)

START = pd.Timestamp('2008-01-01')


def tstat(excess_daily):
    """Newey-West t-stat on mean daily excess return (lag 21 for overlap)."""
    x = excess_daily.dropna().values
    n = len(x)
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, 22):
        w = 1 - l / 22
        cov = (e[l:] @ e[:-l]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return mu / se if se > 0 else np.nan


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    base = build(d, f)

    spy_ret = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0)
    rsp = d['etf']['RSP'].reindex(close.index).ffill()
    rsp_ret = rsp.pct_change(fill_method=None).fillna(0.0)
    rets, _ = delisting_returns(close, 0.0)

    mom = momentum_mask(f, base, close, d=d)
    print(f'Qualifying names on a typical entry date: '
          f'{mom.loc[START:].sum(axis=1).replace(0, np.nan).median():.0f}')

    # ------------------------------------------------ 1. concentration vs noise
    print()
    print('=' * 135)
    print('TEST 1  --  widen the book until arbitrary selection stops mattering.')
    print('=' * 135)
    rows = []
    for max_pos in (10, 20, 30, 50, 75, 100):
        ex_spy, ex_rsp = [], []
        for seed in range(5):
            port, _ = equity_curve(mom, d, rets, max_pos=max_pos, seed=seed)
            port = port.loc[START:]
            s = stats(port, spy_ret.loc[START:], '')
            sr = stats(port, rsp_ret.loc[START:], '')
            ex_spy.append(s['excess_%'])
            ex_rsp.append(s['CAGR_%'] - sr['SPY_CAGR_%'])
        rows.append({'positions': max_pos,
                     'mean_excess_vs_SPY': np.mean(ex_spy),
                     'spread_SPY': max(ex_spy) - min(ex_spy),
                     'mean_excess_vs_RSP': np.mean(ex_rsp),
                     'spread_RSP': max(ex_rsp) - min(ex_rsp)})
    conc = pd.DataFrame(rows)
    print(conc.round(2).to_string(index=False))
    print()
    print('READING: "spread" is the gap between the luckiest and unluckiest draw of')
    print('the same strategy. Where spread is larger than mean excess, the result a')
    print('real portfolio gets is determined by luck, not by the rule.')

    # --------------------------------------------- 2. is the edge persistent?
    print()
    print('=' * 135)
    print('TEST 2  --  a 50-name book (low selection noise): where does the edge come from?')
    print('=' * 135)
    port, _ = equity_curve(mom, d, rets, max_pos=50, seed=0)
    port = port.loc[START:]
    sp = spy_ret.reindex(port.index).fillna(0.0)
    rp = rsp_ret.reindex(port.index).fillna(0.0)

    s = stats(port, sp, 'Momentum, 50 names')
    sr = stats(port, rp, 'Momentum, 50 names')
    print(f"CAGR {s['CAGR_%']:.2f}%   SPY {s['SPY_CAGR_%']:.2f}%   RSP {sr['SPY_CAGR_%']:.2f}%   "
          f"sharpe {s['sharpe']:.2f}   maxDD {s['maxDD_%']:.1f}%")
    print(f"t-stat of daily excess vs SPY: {tstat(port - sp):.2f}    "
          f"vs RSP: {tstat(port - rp):.2f}   (needs ~2.0 to be distinguishable from luck)")

    yr = pd.DataFrame({'s': port, 'spy': sp, 'rsp': rp}).groupby(port.index.year).apply(
        lambda g: pd.Series({'strategy_%': ((1 + g['s']).prod() - 1) * 100,
                             'spy_%': ((1 + g['spy']).prod() - 1) * 100,
                             'rsp_%': ((1 + g['rsp']).prod() - 1) * 100}))
    yr['vs_SPY'] = yr['strategy_%'] - yr['spy_%']
    yr['vs_RSP'] = yr['strategy_%'] - yr['rsp_%']
    print()
    print(yr.round(1).to_string())

    tot = yr['vs_SPY'].sum()
    best2 = yr['vs_SPY'].nlargest(2)
    print()
    print(f"Total excess vs SPY across {len(yr)} years: {tot:+.1f}pp")
    print(f"Best two years ({', '.join(str(i) for i in best2.index)}): {best2.sum():+.1f}pp")
    print(f"The other {len(yr) - 2} years combined:        {tot - best2.sum():+.1f}pp")
    print(f"Years beating SPY: {(yr['vs_SPY'] > 0).sum()}/{len(yr)}   "
          f"beating RSP: {(yr['vs_RSP'] > 0).sum()}/{len(yr)}")

    # --------------------------------------------------- 3. cost of being wrong
    print()
    print('=' * 135)
    print('TEST 3  --  what a realistic personal book (10 names) actually experiences')
    print('=' * 135)
    ex = []
    for seed in range(25):
        p, _ = equity_curve(mom, d, rets, max_pos=10, seed=seed)
        p = p.loc[START:]
        st = stats(p, sp, '')
        sr2 = stats(p, rp, '')
        ex.append({'seed': seed, 'CAGR_%': st['CAGR_%'], 'vs_SPY': st['excess_%'],
                   'vs_RSP': st['CAGR_%'] - sr2['SPY_CAGR_%'], 'maxDD_%': st['maxDD_%']})
    e = pd.DataFrame(ex)
    print(e.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2).to_string())
    print()
    print(f"Draws that beat SPY: {(e['vs_SPY'] > 0).sum()}/25   "
          f"beat RSP: {(e['vs_RSP'] > 0).sum()}/25")

    conc.to_csv(DATA / 'verify_concentration.csv', index=False)
    yr.to_csv(DATA / 'verify_yearly50.csv')
    e.to_csv(DATA / 'verify_10name.csv', index=False)
    print('\nSaved -> research/data/verify_concentration.csv, verify_yearly50.csv, verify_10name.csv')


if __name__ == '__main__':
    main()
