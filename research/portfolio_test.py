"""Portfolio-level test: does a diversified basket of these setups beat buy-and-hold?

Median forward return understates a positively-skewed strategy -- a basket of lottery
tickets can compound well even when most individual positions lose. This builds real
equity curves: rebalance monthly, equal-weight the active book, charge costs, and
compare against SPY and equal-weight RSP.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build, setups

pd.set_option('display.width', 300)

COST_BPS = 20          # round-trip slippage + commission on each position
MAX_POSITIONS = 20


def equity_curve(mask, d, hold_days=126, max_pos=MAX_POSITIONS, cost_bps=COST_BPS):
    """Equal-weight book, entries taken monthly, each held `hold_days`."""
    close = d['close']
    rets = close.pct_change(fill_method=None).fillna(0.0)
    idx = close.index

    # Monthly entry dates (last trading day of each month)
    entry_days = set(pd.Series(idx, index=idx).groupby(idx.to_period('M')).last())

    weights = pd.DataFrame(0.0, index=idx, columns=close.columns)
    open_positions = {}          # ticker -> exit index
    turnover = pd.Series(0.0, index=idx)

    mask_v = mask.values
    cols = list(close.columns)
    col_pos = {c: i for i, c in enumerate(cols)}

    for i, dt in enumerate(idx):
        # close expiring positions
        for t in [t for t, e in open_positions.items() if e <= i]:
            open_positions.pop(t)

        if dt in entry_days:
            # candidates signalled in the last month
            lo = max(0, i - 21)
            recent = mask_v[lo:i + 1].any(axis=0)
            cand = [cols[j] for j in np.where(recent)[0] if cols[j] not in open_positions]
            room = max_pos - len(open_positions)
            if room > 0 and cand:
                # deterministic selection: largest drawdown first is arbitrary, so
                # take alphabetical to avoid smuggling in a hidden factor
                for t in sorted(cand)[:room]:
                    open_positions[t] = min(i + hold_days, len(idx) - 1)

        if open_positions:
            w = 1.0 / len(open_positions)
            for t in open_positions:
                weights.iat[i, col_pos[t]] = w

    # daily portfolio return; unallocated capital earns nothing
    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)

    # cost: charge on weight changes
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    port = port - dw * (cost_bps / 2 / 10000.0)

    return port, weights


def stats(port, bench, label, n_positions=None):
    common = port.index
    eq = (1 + port).cumprod()
    be = (1 + bench.reindex(common).fillna(0.0)).cumprod()
    yrs = (common[-1] - common[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    bcagr = be.iloc[-1] ** (1 / yrs) - 1
    vol = port.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    bdd = (be / be.cummax() - 1).min()
    return {
        'strategy': label,
        'CAGR_%': cagr * 100,
        'SPY_CAGR_%': bcagr * 100,
        'excess_%': (cagr - bcagr) * 100,
        'vol_%': vol * 100,
        'sharpe': (port.mean() * 252) / (vol + 1e-12),
        'maxDD_%': dd * 100,
        'SPY_maxDD_%': bdd * 100,
        'final_x': eq.iloc[-1],
        'SPY_x': be.iloc[-1],
        'avg_positions': n_positions,
    }


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)

    base = build(d, f)
    masks = setups(f, base, d)

    spy = d['etf']['SPY']
    spy_ret = spy.pct_change(fill_method=None).fillna(0.0)
    rsp = d['etf']['RSP'].reindex(d['close'].index).ffill()
    rsp_ret = rsp.pct_change(fill_method=None).fillna(0.0)

    start = pd.Timestamp('2008-01-01')

    rows = []
    curves = {}
    for label, mask in masks.items():
        if mask.values.sum() < 60:
            print(f'skip (too few signals): {label}')
            continue
        port, w = equity_curve(mask, d)
        port = port.loc[start:]
        avg_pos = (w.loc[start:] > 0).sum(axis=1).mean()
        rows.append(stats(port, spy_ret.loc[start:], label, round(avg_pos, 1)))
        curves[label] = port

    print('=' * 175)
    print(f'PORTFOLIO TEST  --  monthly entries, 6-month holds, max {MAX_POSITIONS} '
          f'equal-weight positions, {COST_BPS}bps round trip, from {start.date()}')
    print('=' * 175)
    res = pd.DataFrame(rows).sort_values('CAGR_%', ascending=False)
    print(res.round(2).to_string(index=False))

    # Equal-weight benchmark for reference
    rsp_stats = stats(rsp_ret.loc[start:], spy_ret.loc[start:], 'BENCHMARK: RSP equal-weight')
    print()
    print(pd.DataFrame([rsp_stats]).round(2).to_string(index=False))

    print()
    print('=' * 175)
    print('ROLLING 3-YEAR EXCESS CAGR vs SPY  --  is any edge persistent or episodic?')
    print('=' * 175)
    roll = {}
    for label, port in curves.items():
        ex = port - spy_ret.reindex(port.index).fillna(0.0)
        roll[label[:38]] = ex.rolling(756).mean() * 252 * 100
    rdf = pd.DataFrame(roll).dropna(how='all')
    print(rdf.resample('YE').last().round(1).to_string())

    res.to_csv(DATA / 'portfolio_results.csv', index=False)
    print('\nSaved -> research/data/portfolio_results.csv')


if __name__ == '__main__':
    main()
