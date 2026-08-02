"""Three checks before the ranking factor is allowed into a screen.

recovery_screen.py found the factor works in three eras and inverts in 2022-2026.
That is the era the user cares most about, so it needs an explanation rather than a
footnote. The prime suspect is the benchmark: 2022-2026 was carried by a handful of
mega-cap AI names, so *anything* outside them looks terrible against SPY. Re-running
against the equal-weight index removes that distortion.

Also settles two practical questions: what the rule actually says in plain numbers,
and where Intel itself ranked on the day the position was opened.

Run:  venv/bin/python3 research/recovery_check.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analogues import setup_mask
from engine import DATA

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 70)

ERAS = [('2007-2011', '2007-01-01', '2011-12-31'),
        ('2012-2016', '2012-01-01', '2016-12-31'),
        ('2017-2021', '2017-01-01', '2021-12-31'),
        ('2022-2026', '2022-01-01', '2026-07-31')]


def era_table(mask, factor, ex, title):
    rows = []
    for label, a, b in ERAS:
        sl = slice(pd.Timestamp(a), pd.Timestamp(b))
        m = mask.loc[sl]
        v = factor.loc[sl].where(m).stack().dropna()
        e = ex.loc[sl].where(m).stack().dropna()
        common = v.index.intersection(e.index)
        if len(common) < 300:
            continue
        v, e = v.loc[common], e.loc[common]
        q = pd.qcut(v, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        med = e.groupby(q, observed=True).median()
        rows.append({'era': label, 'n': len(common),
                     **{k: med.get(k, np.nan) for k in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']},
                     'Q5-Q1': med.get('Q5', np.nan) - med.get('Q1', np.nan),
                     'ic': v.corr(e, method='spearman')})
    df = pd.DataFrame(rows)
    print()
    print(title)
    print('-' * len(title))
    print(df.round(2).to_string(index=False))
    return df


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    spy = d['etf']['SPY']
    rsp = d['etf']['RSP'].reindex(close.index).ffill()
    mask = setup_mask(d, f)
    factor = f['off_52w_low_pct']

    h = 252
    fwd = (close.shift(-h) / close - 1.0) * 100
    ex_spy = fwd.sub((spy.shift(-h) / spy - 1.0) * 100, axis=0)
    ex_rsp = fwd.sub((rsp.shift(-h) / rsp - 1.0) * 100, axis=0)

    print('=' * 120)
    print('CHECK 1  --  is the 2022-2026 inversion a benchmark artifact?')
    print('=' * 120)
    print('Median 12-month excess return by quintile of "how far already off the 52w low".')
    era_table(mask, factor, ex_spy, 'Measured against SPY (cap-weighted)')
    e2 = era_table(mask, factor, ex_rsp, 'Measured against RSP (equal-weight)')
    print()
    print(f'Eras pointing the same way vs SPY: 3/4     '
          f'vs RSP: {(e2["Q5-Q1"] > 0).sum()}/{len(e2)}')

    # ------------------------------------------------------------ 2. the plain rule
    print()
    print('=' * 120)
    print('CHECK 2  --  what the rule says in plain numbers')
    print('=' * 120)
    v = factor.where(mask).stack().dropna()
    cuts = v.quantile([0.2, 0.4, 0.6, 0.8]).round(1)
    print('Quintile boundaries for "% above the 52-week low", across the whole setup:')
    for p, c in cuts.items():
        print(f'   {int(p * 100)}th percentile : {c:>6.1f}%')
    print()
    print(f'So "top 20%" means the stock has already risen more than '
          f'{cuts.loc[0.8]:.0f}% off its 52-week low')
    print('while still sitting 40%+ below its multi-year high. It has stopped falling')
    print('and started repairing, but has not yet recovered.')

    print()
    for lbl, e in (('vs SPY', ex_spy), ('vs RSP', ex_rsp)):
        vv = factor.where(mask).stack().dropna()
        ee = e.where(mask).stack().dropna()
        common = vv.index.intersection(ee.index)
        vv, ee = vv.loc[common], ee.loc[common]
        top = ee[vv >= cuts.loc[0.8]]
        rest = ee[vv < cuts.loc[0.8]]
        print(f'{lbl}:  top group median {top.median():+.1f}pp, beat {(top > 0).mean() * 100:.0f}%   |   '
              f'everyone else median {rest.median():+.1f}pp, beat {(rest > 0).mean() * 100:.0f}%')

    # ---------------------------------------------------------------- 3. Intel itself
    print()
    print('=' * 120)
    print('CHECK 3  --  where did Intel itself rank?')
    print('=' * 120)
    # The research mask samples Fridays only, which would exclude Intel on a
    # technicality. A live screen runs every day, so rebuild the same conditions
    # without the weekly sampling.
    member = d['member'].reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)
    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_up = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    daily = member & liquid & close.notna() & (f['age_days'] >= 400) & damaged & ind_up
    rank = factor.where(daily).rank(axis=1, pct=True)

    conds = {'index member': member, 'liquid': liquid,
             'down 40%+ from 3y peak': damaged, 'industry healthy': ind_up}

    for label, date in (('2024-10-31  (still falling)', '2024-10-31'),
                        ('2025-04-08  (the low)', '2025-04-08'),
                        ('2025-08-12  (200dma reclaim)', '2025-08-12'),
                        ('2025-11-28  (later in recovery)', '2025-11-28')):
        ts = close.index[close.index.get_indexer([pd.Timestamp(date)], method='nearest')[0]]
        flagged = bool(daily.loc[ts, 'INTC'])
        off = factor.loc[ts, 'INTC']
        r = rank.loc[ts, 'INTC']
        n = int(daily.loc[ts].sum())
        fwd12, exs = fwd.loc[ts, 'INTC'], ex_spy.loc[ts, 'INTC']
        rtxt = (f'top {100 - r * 100:.0f}% of {n} candidates' if pd.notna(r)
                else 'not flagged')
        f12 = f'{fwd12:>7.1f}%' if pd.notna(fwd12) else '    n/a'
        e12 = f'{exs:>7.1f}pp' if pd.notna(exs) else '    n/a'
        print(f'  {label:<34} flagged: {str(flagged):<5}  off low {off:>6.1f}%  '
              f'{rtxt:<26} next 12m {f12}  vs SPY {e12}')
        if not flagged:
            fails = [k for k, v in conds.items() if not bool(v.loc[ts, 'INTC'])]
            print(f'{"":<36}   failed on: {", ".join(fails) if fails else "n/a"}')

    print()
    print('Read this carefully: it is the sharpest test of the whole idea, because it')
    print('asks whether the screen would have found the one trade it was built from.')


if __name__ == '__main__':
    main()
