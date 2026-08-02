"""Find the historical situations that most resembled Intel, and see what happened.

Two questions, in order:

1. ANALOGUES. Of every large-cap situation since 2007, which ones actually looked
   like Intel did in 2025 -- same drawdown depth, same industry backdrop, same
   lag behind that industry, same size and volatility? What happened to them?
   Nearest-neighbour search in standardised feature space, so the comparison set is
   chosen by resemblance to Intel *before* the outcome is known, not by outcome.

2. DISCRIMINANT. Among all Intel-shaped setups, roughly 40% beat the index. What
   separated those from the rest? Candidate discriminators are chosen on 2007-2017
   and confirmed once on 2018-2026. Anything that only works in-sample is discarded.

This is the honest version of "find more examples like Intel". Picking winners first
and reading a rule off them is what produced the playbook that failed.

Run:  venv/bin/python3 research/analogues.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 70)

TRAIN_END = pd.Timestamp('2017-12-31')
FEATURES = ['dd_3y', 'ret_12m', 'ret_6m', 'rel_12m', 'ind_ret_12m', 'vol_ann',
            'dist_200_pct', 'off_52w_low_pct', 'log_dv']


# ------------------------------------------------------------------ extra features

def extra_features(d, f):
    """Features the earlier study never tested, all trailing-window only."""
    close, vol = d['close'], d['volume']
    g = {}

    # How long since the stock last made a new 52-week low: repair time.
    low52 = close.rolling(252, min_periods=200).min()
    at_low = close <= low52 * 1.02
    idx = np.arange(len(close.index))
    since = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in close.columns:
        a = at_low[c].values
        last = np.where(a, idx, np.nan)
        last = pd.Series(last).ffill().values
        since[c] = idx - last
    g['days_since_low'] = since

    # Volatility contraction: recent range narrowing relative to the past year.
    r = close.pct_change(fill_method=None)
    v20 = r.rolling(20, min_periods=15).std()
    v250 = r.rolling(250, min_periods=200).std()
    g['vol_contraction'] = v20 / v250

    # Accumulation: share of the last 60 sessions that closed up, volume-weighted.
    up = (r > 0).astype(float)
    vw = (up * vol).rolling(60, min_periods=40).sum()
    tot = vol.rolling(60, min_periods=40).sum()
    g['accumulation'] = vw / tot

    # Is the company uniquely broken, or is the whole industry down with it?
    g['dd_vs_industry'] = f['dd_3y'] * 100 - f['ind_ret_12m'].clip(upper=0)

    # Size proxy. No point-in-time share count exists, so dollar volume stands in.
    g['log_dv'] = np.log10(f['dollar_vol'].clip(lower=1))

    # Trend of the industry itself, and the market backdrop.
    g['ind_ret_3m'] = f['ind_ret_3m']
    return g


def setup_mask(d, f):
    base = build(d, f)
    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_up = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    return base & damaged & ind_up


def panel(d, f, g, mask):
    """Flatten the setup into one row per (date, ticker) with features + outcome."""
    close = d['close']
    spy = d['etf']['SPY']
    cols = {}
    for k in FEATURES:
        src = f[k] if k in f else g[k]
        cols[k] = src.where(mask).stack()
    for k in ('days_since_low', 'vol_contraction', 'accumulation',
              'dd_vs_industry', 'ind_ret_3m'):
        cols[k] = g[k].where(mask).stack()
    cols['above_200'] = f['above_200'].where(mask).stack()
    cols['spy_above_200'] = f['spy_above_200'].where(mask).stack()

    for h, n in ((126, 'fwd_6m'), (252, 'fwd_12m')):
        fwd = (close.shift(-h) / close - 1.0) * 100
        b = (spy.shift(-h) / spy - 1.0) * 100
        cols[n] = fwd.where(mask).stack()
        cols[f'ex_{n}'] = fwd.sub(b, axis=0).where(mask).stack()

    df = pd.DataFrame(cols)
    df.index.names = ['date', 'ticker']
    return df.reset_index()


# ----------------------------------------------------------------- 1. analogues

def find_analogues(df, ref_row, k=30, exclude_after='2025-01-01'):
    pool = df[(df['date'] < pd.Timestamp(exclude_after)) & df['ex_fwd_12m'].notna()]
    pool = pool[pool['ticker'] != 'INTC']
    if pool.empty:
        return pool

    X = pool[FEATURES].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = (X - mu) / sd
    z_ref = (ref_row[FEATURES].astype(float) - mu) / sd
    dist = np.sqrt(((Z - z_ref) ** 2).sum(axis=1))

    out = pool.copy()
    out['distance'] = dist.values
    # One entry per ticker: keep each name's closest match only.
    out = out.sort_values('distance').drop_duplicates('ticker').head(k)
    return out


def intc_reference(d, f, g, mask):
    """Intel's own state on the dates the thesis was live."""
    rows = {}
    for label, date in (('2024-10-31 (falling)', '2024-10-31'),
                        ('2025-04-08 (the low)', '2025-04-08'),
                        ('2025-08-12 (200dma reclaim)', '2025-08-12')):
        ts = pd.Timestamp(date)
        if ts not in d['close'].index:
            ts = d['close'].index[d['close'].index.get_indexer([ts], method='nearest')[0]]
        r = {}
        for k in FEATURES:
            src = f[k] if k in f else g[k]
            r[k] = src.loc[ts, 'INTC']
        r['price'] = d['close'].loc[ts, 'INTC']
        r['date'] = ts
        rows[label] = pd.Series(r)
    return rows


# --------------------------------------------------------------- 2. discriminant

def quintile_report(train, feats):
    rows = []
    for k in feats:
        v = train[k].astype(float)
        if v.notna().sum() < 200 or v.nunique() < 10:
            continue
        try:
            q = pd.qcut(v, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except ValueError:
            continue
        gb = train.groupby(q, observed=True)['ex_fwd_12m']
        med = gb.median()
        if len(med) < 5:
            continue
        rows.append({'feature': k, 'Q1': med.iloc[0], 'Q3': med.iloc[2],
                     'Q5': med.iloc[-1], 'Q5-Q1': med.iloc[-1] - med.iloc[0],
                     'beat% Q5': (train[q == 'Q5']['ex_fwd_12m'] > 0).mean() * 100,
                     'beat% Q1': (train[q == 'Q1']['ex_fwd_12m'] > 0).mean() * 100,
                     'ic': v.corr(train['ex_fwd_12m'], method='spearman')})
    return pd.DataFrame(rows).sort_values('Q5-Q1', key=abs, ascending=False)


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    g = extra_features(d, f)
    mask = setup_mask(d, f)
    df = panel(d, f, g, mask)
    df = df.dropna(subset=FEATURES)
    print(f'Intel-shaped observations with complete features: {len(df):,} '
          f'across {df["ticker"].nunique()} companies')

    refs = intc_reference(d, f, g, mask)

    print()
    print('=' * 130)
    print('INTEL, AS THE SCREEN SAW IT')
    print('=' * 130)
    print(pd.DataFrame(refs).T.round(2).to_string())

    for label, ref in refs.items():
        an = find_analogues(df, ref, k=25)
        if an.empty:
            continue
        print()
        print('=' * 130)
        print(f'CLOSEST HISTORICAL ANALOGUES TO INTEL AT {label}')
        print('=' * 130)
        show = an[['date', 'ticker', 'distance', 'dd_3y', 'ret_12m', 'rel_12m',
                   'ind_ret_12m', 'vol_ann', 'fwd_12m', 'ex_fwd_12m']].copy()
        show['date'] = show['date'].dt.date
        show['dd_3y'] = show['dd_3y'] * 100
        print(show.round(1).to_string(index=False))
        print()
        print(f'  median 12m return   : {an["fwd_12m"].median():.1f}%')
        print(f'  median 12m vs SPY   : {an["ex_fwd_12m"].median():.1f}pp')
        print(f'  share beating SPY   : {(an["ex_fwd_12m"] > 0).mean() * 100:.0f}%')
        print(f'  best / worst        : {an["fwd_12m"].max():.0f}% / {an["fwd_12m"].min():.0f}%')

    # ------------------------------------------------------------- discriminant
    dis = df.dropna(subset=['ex_fwd_12m']).copy()
    train = dis[dis['date'] <= TRAIN_END]
    test = dis[dis['date'] > TRAIN_END]
    feats = FEATURES + ['days_since_low', 'vol_contraction', 'accumulation',
                        'dd_vs_industry', 'ind_ret_3m']

    print()
    print('=' * 130)
    print(f'WHAT SEPARATED WINNERS FROM LOSERS   train {len(train):,} obs '
          f'(2007-2017)   test {len(test):,} obs (2018-2026)')
    print('=' * 130)
    print('Median 12-month excess return vs SPY, by feature quintile, TRAIN ONLY.')
    qr = quintile_report(train, feats)
    print(qr.round(2).to_string(index=False))

    print()
    print('Same table on the HOLDOUT. A discriminator is only real if the sign and')
    print('rough size of Q5-Q1 survive here.')
    qt = quintile_report(test, feats)
    merged = qr[['feature', 'Q5-Q1', 'ic']].merge(
        qt[['feature', 'Q5-Q1', 'ic']], on='feature', suffixes=('_train', '_test'))
    merged['same_sign'] = np.sign(merged['Q5-Q1_train']) == np.sign(merged['Q5-Q1_test'])
    print(merged.round(2).to_string(index=False))

    surviving = merged[merged['same_sign'] & (merged['Q5-Q1_train'].abs() > 5)
                       & (merged['Q5-Q1_test'].abs() > 5)]
    print()
    if surviving.empty:
        print('NOTHING SURVIVES. No feature keeps both its sign and a meaningful spread')
        print('out of sample. There is no discriminator here to build a screen on.')
    else:
        print('SURVIVING DISCRIMINATORS:')
        print(surviving.round(2).to_string(index=False))

    df.to_csv(DATA / 'analogue_panel.csv', index=False)
    qr.to_csv(DATA / 'discriminant_train.csv', index=False)
    merged.to_csv(DATA / 'discriminant_holdout.csv', index=False)
    print('\nSaved -> research/data/analogue_panel.csv, discriminant_*.csv')


if __name__ == '__main__':
    main()
