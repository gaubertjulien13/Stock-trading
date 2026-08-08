"""Independent re-check of the conclusions before they go into a live script.

Three untested assumptions in the earlier pipeline could each flip the answer:

1. BENCHMARK. A 20-name equal-weight book was compared against SPY, which is
   cap-weighted. Equal-weighting is itself a bet on smaller names. The honest
   benchmark is RSP.
2. SELECTION. When more names qualify than there are slots, the earlier code
   picked alphabetically. If the reported edge depends on that arbitrary choice,
   it is not an edge.
3. DELISTINGS. Returns were filled with 0 when a ticker stopped trading, so a
   position held into a bankruptcy silently returned 0% instead of -100%.

Run:  venv/bin/python3 research/validate_assumptions.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build, setups
from portfolio_test import stats
from tune_momentum import momentum_mask

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 60)

START = pd.Timestamp('2008-01-01')
COST_BPS = 20


def delisting_returns(close, terminal_ret):
    """Return matrix that charges `terminal_ret` when a ticker stops trading.

    A ticker whose last valid price is before the end of the sample has left the
    data. Some left via buyout (positive), some via bankruptcy (-100%). Setting
    the shock to 0 reproduces the earlier assumption; -1.0 is the worst case.
    """
    rets = close.pct_change(fill_method=None)
    last_valid = close.apply(lambda s: s.last_valid_index())
    end = close.index[-1]
    rets = rets.fillna(0.0)
    n_dead = 0
    for t, lv in last_valid.items():
        if lv is None or pd.isna(lv) or lv >= end:
            continue
        loc = close.index.get_loc(lv)
        if loc + 1 < len(close.index):
            rets.iat[loc + 1, close.columns.get_loc(t)] = terminal_ret
            n_dead += 1
    return rets, n_dead


def equity_curve(mask, d, rets, hold_days=126, max_pos=20, cost_bps=COST_BPS,
                 score=None, seed=None):
    """Equal-weight book, monthly entries, each position held `hold_days`.

    `score` ranks candidates when there are more than there are slots (highest
    first). `seed` picks at random instead. Neither set -> alphabetical, matching
    the earlier code.
    """
    close = d['close']
    idx = close.index
    entry_days = set(pd.Series(idx, index=idx).groupby(idx.to_period('M')).last())

    weights = pd.DataFrame(0.0, index=idx, columns=close.columns)
    open_positions = {}
    mask_v = mask.values
    cols = list(close.columns)
    col_pos = {c: i for i, c in enumerate(cols)}
    score_v = score.values if score is not None else None
    rng = np.random.default_rng(seed) if seed is not None else None

    for i, dt in enumerate(idx):
        for t in [t for t, e in open_positions.items() if e <= i]:
            open_positions.pop(t)

        if dt in entry_days:
            lo = max(0, i - 21)
            recent = mask_v[lo:i + 1].any(axis=0)
            cand_j = [j for j in np.where(recent)[0] if cols[j] not in open_positions]
            room = max_pos - len(open_positions)
            if room > 0 and cand_j:
                if score_v is not None:
                    vals = score_v[i, cand_j]
                    vals = np.where(np.isfinite(vals), vals, -np.inf)
                    order = np.argsort(-vals)
                    pick = [cand_j[k] for k in order[:room]]
                elif rng is not None:
                    pick = list(rng.permutation(cand_j)[:room])
                else:
                    pick = sorted(cand_j, key=lambda j: cols[j])[:room]
                for j in pick:
                    open_positions[cols[j]] = min(i + hold_days, len(idx) - 1)

        if open_positions:
            w = 1.0 / len(open_positions)
            for t in open_positions:
                weights.iat[i, col_pos[t]] = w

    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    port = port - dw * (cost_bps / 2 / 10000.0)
    return port, weights


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    base = build(d, f)

    spy_ret = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0)
    rsp = d['etf']['RSP'].reindex(close.index).ffill()
    rsp_ret = rsp.pct_change(fill_method=None).fillna(0.0)

    rets0, n_dead = delisting_returns(close, 0.0)       # earlier assumption
    rets_bk, _ = delisting_returns(close, -1.0)         # worst case

    print('=' * 130)
    print('CHECK 0  --  how much delisting risk is actually in the data?')
    print('=' * 130)
    print(f'Tickers whose price series ends before {close.index[-1].date()}: {n_dead}')

    mom = momentum_mask(f, base, close, d=d)
    mom_score = (close.shift(21) / close.shift(21 + 252) - 1.0).where(base)
    masks = setups(f, base, d)
    intel = masks['A. INTEL playbook (damaged + hot industry + reclaim)']
    laggard = masks['C. Damaged + reclaim (industry agnostic)']
    laggard_j = masks['J. Laggard turning: damaged + hot industry + rel_3m>0']

    # ---------------------------------------------------------------- benchmarks
    print()
    print('=' * 130)
    print('CHECK 1  --  the right benchmark. Equal-weight RSP vs cap-weight SPY.')
    print('=' * 130)
    b = stats(rsp_ret.loc[START:], spy_ret.loc[START:], 'RSP equal-weight (benchmark)')
    print(pd.DataFrame([b]).drop(columns=['avg_positions']).round(2).to_string(index=False))

    # ------------------------------------------------- selection-rule robustness
    print()
    print('=' * 130)
    print('CHECK 2  --  does the momentum edge survive a different selection rule?')
    print('=' * 130)
    rows = []
    variants = [('alphabetical (as tested before)', dict()),
                ('highest momentum first', dict(score=mom_score))]
    for s in range(5):
        variants.append((f'random seed {s}', dict(seed=s)))

    curves = {}
    for name, kw in variants:
        port, w = equity_curve(mom, d, rets0, **kw)
        port = port.loc[START:]
        curves[name] = port
        st = stats(port, spy_ret.loc[START:], name)
        st_rsp = stats(port, rsp_ret.loc[START:], name)
        rows.append({'selection': name, 'CAGR_%': st['CAGR_%'],
                     'excess_vs_SPY': st['excess_%'],
                     'excess_vs_RSP': st['CAGR_%'] - st_rsp['SPY_CAGR_%'],
                     'sharpe': st['sharpe'], 'maxDD_%': st['maxDD_%']})
    rnd = pd.DataFrame(rows)
    print(rnd.round(2).to_string(index=False))
    r = rnd[rnd['selection'].str.startswith('random')]
    print(f"\nRandom-selection excess vs SPY: mean {r['excess_vs_SPY'].mean():.2f}%  "
          f"range [{r['excess_vs_SPY'].min():.2f}, {r['excess_vs_SPY'].max():.2f}]")
    print(f"Random-selection excess vs RSP: mean {r['excess_vs_RSP'].mean():.2f}%  "
          f"range [{r['excess_vs_RSP'].min():.2f}, {r['excess_vs_RSP'].max():.2f}]")

    # ---------------------------------------------------------- delisting shock
    print()
    print('=' * 130)
    print('CHECK 3  --  charge -100% on every delisting instead of 0%.')
    print('=' * 130)
    rows = []
    for label, m, sc in [('Momentum top20%', mom, mom_score),
                         ('A. Intel playbook', intel, None),
                         ('C. Damaged + reclaim', laggard, None),
                         ('J. Laggard turning', laggard_j, None)]:
        if m.values.sum() < 60:
            rows.append({'strategy': label, 'note': 'too few signals'})
            continue
        for tag, rr in [('delist=0%', rets0), ('delist=-100%', rets_bk)]:
            port, _ = equity_curve(m, d, rr, score=sc)
            port = port.loc[START:]
            st = stats(port, spy_ret.loc[START:], label)
            st_r = stats(port, rsp_ret.loc[START:], label)
            rows.append({'strategy': label, 'delisting': tag, 'CAGR_%': st['CAGR_%'],
                         'excess_vs_SPY': st['excess_%'],
                         'excess_vs_RSP': st['CAGR_%'] - st_r['SPY_CAGR_%'],
                         'sharpe': st['sharpe'], 'maxDD_%': st['maxDD_%']})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ------------------------------------------------- momentum, decade by decade
    print()
    print('=' * 130)
    print('CHECK 4  --  momentum finalist, year by year, vs BOTH benchmarks')
    print('=' * 130)
    port = curves['highest momentum first']
    sp = spy_ret.reindex(port.index).fillna(0.0)
    rp = rsp_ret.reindex(port.index).fillna(0.0)
    yr = pd.DataFrame({'s': port, 'spy': sp, 'rsp': rp}).groupby(port.index.year).apply(
        lambda g: pd.Series({'strategy_%': ((1 + g['s']).prod() - 1) * 100,
                             'spy_%': ((1 + g['spy']).prod() - 1) * 100,
                             'rsp_%': ((1 + g['rsp']).prod() - 1) * 100}))
    yr['vs_SPY'] = yr['strategy_%'] - yr['spy_%']
    yr['vs_RSP'] = yr['strategy_%'] - yr['rsp_%']
    print(yr.round(1).to_string())
    print(f"\nYears beating SPY: {(yr['vs_SPY'] > 0).sum()}/{len(yr)}   "
          f"beating RSP: {(yr['vs_RSP'] > 0).sum()}/{len(yr)}")
    print(f"Median annual excess  vs SPY {yr['vs_SPY'].median():.1f}%   "
          f"vs RSP {yr['vs_RSP'].median():.1f}%")

    out = pd.DataFrame({'momentum': (1 + port).cumprod(),
                        'spy': (1 + sp).cumprod(), 'rsp': (1 + rp).cumprod()})
    out.to_csv(DATA / 'verify_equity.csv')
    yr.to_csv(DATA / 'verify_yearly.csv')
    rnd.to_csv(DATA / 'verify_selection.csv', index=False)
    print('\nSaved -> research/data/verify_*.csv')


if __name__ == '__main__':
    main()
