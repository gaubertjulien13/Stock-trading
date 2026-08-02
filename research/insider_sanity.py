"""Sanity check: is the insider pipeline wired correctly, or just returning zeros?

A null result is only worth reporting if the measurement works. Insider buying has a
well-documented *short-horizon* announcement effect -- a small abnormal return in the
days and weeks right after the Form 4 hits the tape. If that shows up here, the
plumbing is sound and the longer-horizon null is a real finding. If even the
announcement effect is missing, the pipeline is broken and nothing else can be trusted.

Run:  venv/bin/python3 research/insider_sanity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA, assign_industry
from insider_test import build_matrices, clean_filings
from smid_test import build_features, load_prices

pd.set_option('display.width', 240)


def main():
    d = load_prices()
    industry = assign_industry(d['close'], d['etf'])
    f = build_features(d, industry)
    close = d['close']

    df = clean_filings(close)
    n_ins, usd = build_matrices(df, close)

    member = pd.read_parquet(DATA / 'sp500_membership.parquet')
    member = member.reindex(close.index.union(member.index)).ffill()
    member = member.reindex(close.index).fillna(False).astype(bool)
    member = member.reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)
    base = member & liquid & close.notna() & (f['age_days'] >= 400)
    start = pd.Timestamp('2007-01-01')

    rets = close.pct_change(fill_method=None)
    ew = rets.where(base).mean(axis=1).fillna(0.0)
    ew_px = (1 + ew).cumprod()

    # Event day = the first day a 90-day window goes from no buying to buying,
    # i.e. the arrival of fresh news rather than the stale state of the window.
    fresh = (n_ins.diff() > 0) & base

    print()
    print('=' * 110)
    print('ANNOUNCEMENT EFFECT  --  return after a Form 4 purchase filing hits the tape')
    print('=' * 110)
    print('S&P 500, 2007-2026. Excess = return minus an equal-weight index of the universe.')
    print()

    rows = []
    for h in (1, 2, 3, 5, 10, 21, 42, 63, 126, 252):
        fwd = ((close.shift(-h) / close - 1.0) * 100).loc[start:]
        b = ((ew_px.shift(-h) / ew_px - 1.0) * 100).loc[start:]
        m = fresh.loc[start:]
        ex = fwd.sub(b, axis=0).where(m).stack().dropna()
        if len(ex) < 200:
            continue
        # t-stat on the mean; events overlap little at short horizons
        t = ex.mean() / (ex.std() / np.sqrt(len(ex)))
        rows.append({'days after filing': h, 'n events': len(ex),
                     'mean excess %': ex.mean(), 'median excess %': ex.median(),
                     'beat peers %': (ex > 0).mean() * 100, 't-stat': t})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    print()
    print('READING: a positive, statistically significant mean excess return in the first')
    print('few days confirms the filings, dates and price alignment are correct.')

    # Placebo: the same measurement on random days should show nothing.
    rng = np.random.default_rng(0)
    placebo = pd.DataFrame(rng.random(base.shape) < 0.002, index=base.index,
                           columns=base.columns) & base
    print()
    print('PLACEBO  --  identical measurement on random days (should be ~0):')
    rows = []
    for h in (5, 21, 126):
        fwd = ((close.shift(-h) / close - 1.0) * 100).loc[start:]
        b = ((ew_px.shift(-h) / ew_px - 1.0) * 100).loc[start:]
        ex = fwd.sub(b, axis=0).where(placebo.loc[start:]).stack().dropna()
        t = ex.mean() / (ex.std() / np.sqrt(len(ex)))
        rows.append({'days after': h, 'n': len(ex), 'mean excess %': ex.mean(),
                     't-stat': t})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    res.to_csv(DATA / 'insider_announcement.csv', index=False)
    print('\nSaved -> research/data/insider_announcement.csv')


if __name__ == '__main__':
    main()
