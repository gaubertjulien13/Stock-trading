"""Stress-test the one discriminator that survived, and see if it can rank candidates.

analogues.py found that of fourteen candidate features measured inside the Intel
setup, thirteen flip sign out of sample. One does not: how far the stock has already
risen off its 52-week low. Before that becomes the basis of a ranking algorithm it has
to clear three bars:

1. Does it hold era by era, or only on average across a lucky split?
2. The outcome distribution is violently right-skewed -- a few names triple while the
   median lags. Does the factor pick up the skew, or just shift the median?
3. Ranked into a shortlist, does the top group actually beat the rest of the setup?

Bar 3 is the real question. The user picks the trades; the algorithm only has to sort
candidates better than chance, which is a lower bar than beating the index outright.

Run:  venv/bin/python3 research/recovery_screen.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analogues import extra_features, setup_mask
from engine import DATA
from portfolio_test import stats
from verify import delisting_returns, equity_curve

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 70)

ERAS = [('2007-2011', '2007-01-01', '2011-12-31'),
        ('2012-2016', '2012-01-01', '2016-12-31'),
        ('2017-2021', '2017-01-01', '2021-12-31'),
        ('2022-2026', '2022-01-01', '2026-07-31')]


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    g = extra_features(d, f)
    close = d['close']
    spy = d['etf']['SPY']
    mask = setup_mask(d, f)

    h = 252
    fwd = (close.shift(-h) / close - 1.0) * 100
    bench = (spy.shift(-h) / spy - 1.0) * 100
    ex = fwd.sub(bench, axis=0)

    factor = g['off_52w_low_pct'] if 'off_52w_low_pct' in g else f['off_52w_low_pct']

    # -------------------------------------------------------------- 1. era stability
    print('=' * 120)
    print('BAR 1  --  does "already off the low" hold in every era?')
    print('=' * 120)
    print('Median 12-month excess return vs SPY, by quintile of how far the stock has')
    print('already risen off its 52-week low. Q1 = still scraping the bottom.')
    print()
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
        row = {'era': label, 'n': len(common)}
        row.update({k: med.get(k, np.nan) for k in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']})
        row['Q5-Q1'] = med.get('Q5', np.nan) - med.get('Q1', np.nan)
        row['ic'] = v.corr(e, method='spearman')
        rows.append(row)
    era = pd.DataFrame(rows)
    print(era.round(2).to_string(index=False))
    consistent = (era['Q5-Q1'] > 0).sum()
    print(f'\nEras where the factor points the same way: {consistent}/{len(era)}')

    # --------------------------------------------------------------- 2. the skew
    print()
    print('=' * 120)
    print('BAR 2  --  the outcome distribution, and whether the factor captures the skew')
    print('=' * 120)
    v_all = factor.where(mask).stack().dropna()
    e_all = ex.where(mask).stack().dropna()
    common = v_all.index.intersection(e_all.index)
    v_all, e_all = v_all.loc[common], e_all.loc[common]
    q = pd.qcut(v_all, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    rows = []
    for lbl in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        e = e_all[q == lbl]
        rows.append({'quintile': lbl, 'n': len(e), 'median': e.median(),
                     'mean': e.mean(), 'beat SPY %': (e > 0).mean() * 100,
                     'p10': e.quantile(0.10), 'p90': e.quantile(0.90),
                     '>+50pp %': (e > 50).mean() * 100,
                     '<-30pp %': (e < -30).mean() * 100})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print()
    print(f'Whole setup: median {e_all.median():.1f}pp, mean {e_all.mean():.1f}pp, '
          f'{(e_all > 0).mean() * 100:.0f}% beat SPY, '
          f'{(e_all > 100).mean() * 100:.1f}% more than doubled the index.')
    print('A mean far above the median is the signature of a lottery distribution:')
    print('most tickets lose, a few pay enormously.')

    # ------------------------------------------------- 3. does ranking build a book?
    print()
    print('=' * 120)
    print('BAR 3  --  ranked into a shortlist, does the top group beat the rest?')
    print('=' * 120)
    rets, _ = delisting_returns(close, 0.0)
    spy_ret = spy.pct_change(fill_method=None).fillna(0.0)
    start = pd.Timestamp('2008-01-01')

    rank = factor.where(mask).rank(axis=1, pct=True)
    books = {
        'Intel setup, unranked': mask,
        'Intel setup, top 20% off the low': mask & (rank >= 0.80),
        'Intel setup, top 40% off the low': mask & (rank >= 0.60),
        'Intel setup, bottom 20% (still at lows)': mask & (rank <= 0.20),
    }

    rows = []
    for label, m in books.items():
        if m.loc[start:].values.sum() < 100:
            continue
        cs, dds = [], []
        for seed in range(5):
            p, _ = equity_curve(m, d, rets, max_pos=20, seed=seed)
            p = p.loc[start:]
            s = stats(p, spy_ret.loc[start:], label)
            cs.append(s['CAGR_%'])
            dds.append(s['maxDD_%'])
        rows.append({'book': label, 'signals': int(m.loc[start:].values.sum()),
                     'CAGR mean': np.mean(cs), 'best': max(cs), 'worst': min(cs),
                     'luck spread': max(cs) - min(cs), 'maxDD': np.mean(dds)})
    res = pd.DataFrame(rows)
    spy_c = stats(spy_ret.loc[start:], spy_ret.loc[start:], '')['CAGR_%']
    print(f'SPY over the same window: {spy_c:.2f}% CAGR')
    print(res.round(2).to_string(index=False))

    print()
    print('=' * 120)
    print('VERDICT')
    print('=' * 120)
    top = res[res['book'].str.contains('top 20%')]
    unr = res[res['book'].str.contains('unranked')]
    if not top.empty and not unr.empty:
        lift = top['CAGR mean'].iloc[0] - unr['CAGR mean'].iloc[0]
        spread = top['luck spread'].iloc[0]
        print(f'Ranking lifts the book by {lift:+.2f}% a year, against a luck spread of '
              f'{spread:.2f}pp.')
        print(f'Top-ranked book vs SPY: {top["CAGR mean"].iloc[0] - spy_c:+.2f}% a year.')

    era.to_csv(DATA / 'recovery_era.csv', index=False)
    res.to_csv(DATA / 'recovery_books.csv', index=False)
    print('\nSaved -> research/data/recovery_era.csv, recovery_books.csv')


if __name__ == '__main__':
    main()
