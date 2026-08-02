"""Measure which features predict forward excess return, instead of guessing rules.

Restricts to the "damaged company in a healthy industry" universe -- the setup the
INTC thesis describes -- then, for every candidate day, buckets each feature into
quintiles and measures the forward 6-month return relative to SPY. A feature only
earns a place in the strategy if its quintiles are monotonic and the spread is large
enough to survive out of sample.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

HORIZON = 126


def clean_prices(d):
    """Drop implausible series/segments produced by delisted-ticker data errors."""
    close = d['close'].copy()
    close[close <= 0.01] = np.nan

    rets = close.pct_change(fill_method=None)
    # A >150% one-day move in a large-cap is a data artifact, not a trade.
    bad_mask = rets.abs() > 1.5
    bad_cols = bad_mask.any()[bad_mask.any()].index.tolist()
    for c in bad_cols:
        first_bad = bad_mask[c].idxmax()
        close.loc[first_bad:, c] = np.nan
    d = dict(d)
    d['close'] = close
    for k in ('open', 'high', 'low'):
        m = d[k].copy()
        m[m <= 0.01] = np.nan
        m = m.where(close.notna())
        d[k] = m
    print(f'Cleaned {len(bad_cols)} corrupted series: {bad_cols}')
    return d


def main():
    with open(DATA / 'features.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    d = clean_prices(d)
    close = d['close']
    spy = d['etf']['SPY']

    valid = close.notna()
    fwd = (close.shift(-HORIZON) / close - 1.0) * 100
    spy_fwd = (spy.shift(-HORIZON) / spy - 1.0) * 100
    excess = fwd.sub(spy_fwd, axis=0)
    fwd = fwd.where(valid)
    excess = excess.where(valid)

    member = d['member'].reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)

    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_ok = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)

    weekly = pd.Series(False, index=close.index)
    weekly[close.index.dayofweek == 4] = True
    wk = pd.DataFrame(np.repeat(weekly.values[:, None], close.shape[1], axis=1),
                      index=close.index, columns=close.columns)

    base = member & liquid & valid & wk
    cond = base & damaged & ind_ok

    print()
    print('=' * 110)
    print('UNIVERSE SIZES (weekly stock-observations)')
    print('=' * 110)
    print(f'  index members, liquid                : {int(base.values.sum()):>8,}')
    print(f'  + damaged (dd<=-40%, stale peak)     : {int((base & damaged).values.sum()):>8,}')
    print(f'  + industry healthy                   : {int(cond.values.sum()):>8,}')

    def stats(mask, label):
        e = excess.where(mask).stack().dropna()
        r = fwd.where(mask).stack().dropna()
        return dict(label=label, n=len(e), mean_excess=e.mean(), median_excess=e.median(),
                    beat_spy_pct=(e > 0).mean() * 100, median_ret=r.median(),
                    win_pct=(r > 0).mean() * 100)

    print()
    print('=' * 110)
    print('DOES THE SETUP ITSELF HAVE AN EDGE?  (forward 6m, vs SPY)')
    print('=' * 110)
    rows = [
        stats(base, 'All liquid index members'),
        stats(base & damaged, 'Damaged only'),
        stats(base & ind_ok, 'Healthy industry only'),
        stats(cond, 'Damaged + healthy industry (the setup)'),
    ]
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ------------------------------------------------------------- factor quintiles
    print()
    print('=' * 110)
    print('FACTOR POWER WITHIN THE SETUP  --  forward 6m excess return by quintile')
    print('=' * 110)
    print('A usable factor is monotonic from Q1 to Q5 with a wide spread.')

    candidates = {
        'dd_3y': f['dd_3y'] * 100,
        'rel_12m': f['rel_12m'],
        'rel_3m': f['rel_3m'],
        'ret_12m': f['ret_12m'],
        'ret_6m': f['ret_6m'],
        'ret_3m': f['ret_3m'],
        'ret_1m': f['ret_1m'],
        'dist_200_pct': f['dist_200_pct'],
        'sma200_slope': f['sma200_slope'],
        'off_52w_low_pct': f['off_52w_low_pct'],
        'vol_ann': f['vol_ann'],
        'ind_ret_12m': f['ind_ret_12m'],
        'ind_ret_3m': f['ind_ret_3m'],
        'peak_age_days': f['peak_age_days'],
    }

    e_flat = excess.where(cond).stack().dropna()
    results = []
    for name, mat in candidates.items():
        v = mat.where(cond).stack().dropna()
        common = v.index.intersection(e_flat.index)
        if len(common) < 500:
            continue
        v, e = v.loc[common], e_flat.loc[common]
        try:
            q = pd.qcut(v, 5, labels=['Q1 low', 'Q2', 'Q3', 'Q4', 'Q5 high'], duplicates='drop')
        except ValueError:
            continue
        g = e.groupby(q, observed=True).agg(['mean', 'median', 'size'])
        row = {'factor': name, 'n': len(common)}
        for lbl in g.index:
            row[str(lbl)] = g.loc[lbl, 'median']
        row['spread_Q5_Q1'] = g['median'].iloc[-1] - g['median'].iloc[0]
        row['ic_spearman'] = v.corr(e, method='spearman')
        results.append(row)

    res = pd.DataFrame(results).sort_values('ic_spearman', key=abs, ascending=False)
    print()
    print(res.round(2).to_string(index=False))

    print()
    print('=' * 110)
    print('READING: positive spread = higher factor value predicts better performance.')
    print('|spearman| below ~0.05 is noise at this sample size.')
    print('=' * 110)

    # ------------------------------------------- stability of the strongest factors
    print()
    print('STABILITY CHECK -- spearman IC by era for the top factors')
    top = res.reindex(res['ic_spearman'].abs().sort_values(ascending=False).index)['factor'].head(6)
    eras = [('2007-2012', '2007-01-01', '2012-12-31'),
            ('2013-2018', '2013-01-01', '2018-12-31'),
            ('2019-2022', '2019-01-01', '2022-12-31'),
            ('2023-2026', '2023-01-01', '2026-12-31')]
    rows = []
    for name in top:
        mat = candidates[name]
        row = {'factor': name}
        for lbl, a, b in eras:
            sl = slice(pd.Timestamp(a), pd.Timestamp(b))
            v = mat.loc[sl].where(cond.loc[sl]).stack().dropna()
            e = excess.loc[sl].where(cond.loc[sl]).stack().dropna()
            common = v.index.intersection(e.index)
            row[lbl] = v.loc[common].corr(e.loc[common], method='spearman') if len(common) > 200 else np.nan
            row[f'n_{lbl}'] = len(common)
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    with open(DATA / 'clean.pkl', 'wb') as fh:
        pickle.dump((d, f), fh)
    print('\nSaved cleaned data -> research/data/clean.pkl')


if __name__ == '__main__':
    main()
