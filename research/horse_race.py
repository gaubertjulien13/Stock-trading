"""Honest horse race of candidate setups at the weeks-to-months horizon.

The Intel playbook failed validation, so rather than tune a losing idea, this tests a
spread of structurally different hypotheses -- including the direct opposite of the
Intel logic -- against the same benchmarks, eras and horizons.

Benchmarks: SPY (cap-weighted, the real alternative use of capital) and RSP
(equal-weighted, which isolates stock selection from mega-cap concentration).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 80)

HORIZONS = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}


def build(d, f):
    close = d['close']
    valid = close.notna()
    member = d['member'].reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)
    weekly = pd.DataFrame(
        np.repeat((close.index.dayofweek == 4)[:, None], close.shape[1], axis=1),
        index=close.index, columns=close.columns)
    base = member & liquid & valid & weekly & (f['age_days'] >= 400)
    return base


def setups(f, base, d):
    close = d['close']
    ind_up = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    ind_down = (~f['ind_above_200'].fillna(True)) | (f['ind_ret_12m'] < 0)
    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)

    above200 = f['above_200']
    reclaim = above200 & (~above200).rolling(63, min_periods=1).min().astype(bool).shift(1).fillna(False)

    hi52 = close.rolling(252, min_periods=200).max()
    near_high = close >= hi52 * 0.95

    rising200 = f['sma200_slope'] > 0

    # cross-sectional momentum rank, computed per date over the eligible universe
    r12 = f['ret_12m'].where(base)
    mom_rank = r12.rank(axis=1, pct=True)

    rsi_proxy = f['ret_1m']

    return {
        'A. INTEL playbook (damaged + hot industry + reclaim)':
            base & damaged & ind_up & reclaim,
        'B. Damaged + WEAK industry + reclaim (cyclical washout)':
            base & damaged & ind_down & reclaim,
        'C. Damaged + reclaim (industry agnostic)':
            base & damaged & reclaim,
        'D. Pure 12m momentum (top 20%) + above 200dma':
            base & (mom_rank >= 0.80) & above200,
        'E. Momentum + rising 200dma + hot industry':
            base & (mom_rank >= 0.80) & above200 & rising200 & ind_up,
        'F. Near 52w high + hot industry':
            base & near_high & ind_up,
        'G. Buy the dip in an uptrend (1m pullback, rising 200dma)':
            base & above200 & rising200 & (rsi_proxy <= -5),
        'H. Quality trend: above 200dma + rising 200dma':
            base & above200 & rising200,
        'I. Deep value bounce: dd<=-60% + reclaim':
            base & (f['dd_3y'] <= -0.60) & (f['peak_age_days'] >= 126) & reclaim,
        'J. Laggard turning: damaged + hot industry + rel_3m>0':
            base & damaged & ind_up & (f['rel_3m'] > 0) & above200,
        'Z. BASELINE: all eligible members':
            base,
    }


def evaluate(masks, d, f, out_csv=None):
    close = d['close']
    spy = d['etf']['SPY']
    rsp = d['etf']['RSP'].reindex(close.index).ffill()

    rows = []
    for label, mask in masks.items():
        row = {'setup': label, 'n_signals': int(mask.values.sum())}
        for hname, h in HORIZONS.items():
            fwd = (close.shift(-h) / close - 1.0) * 100
            bench = (spy.shift(-h) / spy - 1.0) * 100
            benchr = (rsp.shift(-h) / rsp - 1.0) * 100
            ex = fwd.sub(bench, axis=0).where(mask).stack().dropna()
            exr = fwd.sub(benchr, axis=0).where(mask).stack().dropna()
            r = fwd.where(mask).stack().dropna()
            if len(r) < 100:
                continue
            row[f'{hname}_med_ret'] = r.median()
            row[f'{hname}_med_exSPY'] = ex.median()
            row[f'{hname}_beatSPY%'] = (ex > 0).mean() * 100
            row[f'{hname}_med_exRSP'] = exr.median()
        rows.append(row)
    return pd.DataFrame(rows)


def by_era(masks, d, h=126):
    close = d['close']
    spy = d['etf']['SPY']
    fwd = (close.shift(-h) / close - 1.0) * 100
    bench = (spy.shift(-h) / spy - 1.0) * 100
    ex = fwd.sub(bench, axis=0)

    eras = [('2007-2012', '2007-01-01', '2012-12-31'),
            ('2013-2018', '2013-01-01', '2018-12-31'),
            ('2019-2022', '2019-01-01', '2022-12-31'),
            ('2023-2026', '2023-01-01', '2026-12-31')]
    rows = []
    for label, mask in masks.items():
        row = {'setup': label}
        for ename, a, b in eras:
            sl = slice(pd.Timestamp(a), pd.Timestamp(b))
            e = ex.loc[sl].where(mask.loc[sl]).stack().dropna()
            row[ename] = e.median() if len(e) >= 100 else np.nan
            row[f'n_{ename[:4]}'] = len(e)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)

    base = build(d, f)
    masks = setups(f, base, d)

    print('=' * 190)
    print('HORSE RACE  --  median forward return, and median excess vs SPY / equal-weight RSP')
    print('=' * 190)
    res = evaluate(masks, d, f)
    print(res.round(2).to_string(index=False))

    print()
    print('=' * 190)
    print('STABILITY  --  median 6m excess return vs SPY, by era')
    print('=' * 190)
    era = by_era(masks, d)
    print(era.round(2).to_string(index=False))

    res.to_csv(DATA / 'horse_race.csv', index=False)
    era.to_csv(DATA / 'horse_race_era.csv', index=False)
    print('\nSaved -> research/data/horse_race.csv')


if __name__ == '__main__':
    main()
