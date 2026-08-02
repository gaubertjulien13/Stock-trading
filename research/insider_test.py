"""Does insider buying predict returns on a weeks-to-months horizon?

The last signal testable without paid data, and the only genuinely point-in-time
one: a Form 4 has a filing date, and nobody outside the company could act before it.

Two data hazards handled up front:

1. Filers enter garbage. Reported price-per-share runs up to $250,000,000 in the raw
   data. Every purchase is therefore re-valued at the market close on the filing
   date, and records whose reported price is wildly inconsistent with the market are
   dropped rather than trusted.
2. Look-ahead. Signals use FILING_DATE, never the transaction date, and positions
   fill at the next session's open.

Benchmark is an equal-weight index of the same eligible universe -- what picking at
random from the same pool gives.

Run:  venv/bin/python3 research/insider_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from smid_test import build_features, cagr, equity, load_prices
from engine import assign_industry

pd.set_option('display.width', 280)
pd.set_option('display.max_columns', 60)

UNIVERSES = {
    'S&P 500 large caps': dict(member='sp500_membership.parquet', start='2007-01-01',
                               min_dv=20e6),
    'S&P 400 mid caps': dict(member='sp400_membership.parquet', start='2013-01-01',
                             min_dv=10e6),
    'S&P 600 small caps': dict(member='sp600_membership.parquet', start='2020-07-01',
                               min_dv=3e6),
}

LOOKBACK = 90       # calendar days over which insider buying is accumulated


def clean_filings(close):
    """Re-value every purchase at the market close on its filing date."""
    df = pd.read_parquet(DATA / 'insider_buys.parquet')
    n0 = len(df)
    df = df[df['ticker'].isin(close.columns)]
    n1 = len(df)

    idx = close.index
    pos = idx.searchsorted(df['filing_date'].values, side='left')
    ok = pos < len(idx)
    df = df[ok].copy()
    df['sig_date'] = idx[pos[ok]]

    px = close.stack().rename('mkt')
    df = df.join(px, on=['sig_date', 'ticker'])
    df = df.dropna(subset=['mkt'])
    n2 = len(df)

    df['reported_px'] = df['value'] / df['shares']
    ratio = df['reported_px'] / df['mkt']
    sane = ratio.between(0.2, 5.0)
    df = df[sane].copy()
    n3 = len(df)

    df['usd'] = df['shares'] * df['mkt']

    print('=' * 100)
    print('FORM 4 DATA CLEANING')
    print('=' * 100)
    print(f'  raw purchase rows (ticker x filing date) : {n0:>9,}')
    print(f'  ticker matches a name we have prices for : {n1:>9,}')
    print(f'  price available on the filing date       : {n2:>9,}')
    print(f'  reported price consistent with market    : {n3:>9,}  '
          f'({100 * n3 / max(n2, 1):.1f}% kept)')
    print(f'  total value after re-pricing             : ${df["usd"].sum() / 1e9:,.1f}B')
    print(f'  median purchase                          : ${df["usd"].median():,.0f}')
    print(f'  date range                               : {df["sig_date"].min().date()} .. '
          f'{df["sig_date"].max().date()}')
    return df


def build_matrices(df, close):
    """Trailing-90-day insider buying, as (date x ticker) matrices."""
    cols = close.columns
    ins = pd.DataFrame(0.0, index=close.index, columns=cols)
    usd = pd.DataFrame(0.0, index=close.index, columns=cols)

    piv_i = df.pivot_table(index='sig_date', columns='ticker', values='n_insiders',
                           aggfunc='sum')
    piv_v = df.pivot_table(index='sig_date', columns='ticker', values='usd',
                           aggfunc='sum')
    ins.update(piv_i.reindex(index=close.index, columns=cols))
    usd.update(piv_v.reindex(index=close.index, columns=cols))

    win = int(LOOKBACK * 252 / 365)
    return (ins.rolling(win, min_periods=1).sum(),
            usd.rolling(win, min_periods=1).sum())


def evaluate(name, cfg, d, f, n_ins, usd):
    close = d['close']
    start = pd.Timestamp(cfg['start'])
    member = pd.read_parquet(DATA / cfg['member'])
    member = member.reindex(close.index.union(member.index)).ffill()
    member = member.reindex(close.index).fillna(False).astype(bool)
    member = member.reindex(columns=close.columns).fillna(False)

    liquid = (f['dollar_vol'] >= cfg['min_dv']) & (close >= 5.0)
    base_full = member & liquid & close.notna() & (f['age_days'] >= 400)
    weekly = pd.DataFrame(np.repeat((close.index.dayofweek == 4)[:, None],
                                    close.shape[1], axis=1),
                          index=close.index, columns=close.columns)
    base = base_full & weekly

    # buying scaled by how much the stock normally trades
    intensity = (usd / f['dollar_vol']).replace([np.inf, -np.inf], np.nan)

    any_buy = n_ins >= 1
    cluster = n_ins >= 3
    big_cluster = n_ins >= 5
    heavy = intensity >= 0.5          # half a day's dollar volume bought
    very_heavy = intensity >= 2.0

    setups = {
        'Any insider buying (90d)': base & any_buy,
        'Cluster: 3+ insiders (90d)': base & cluster,
        'Cluster: 5+ insiders (90d)': base & big_cluster,
        'Heavy: >0.5 day of volume bought': base & heavy,
        'Very heavy: >2 days of volume': base & very_heavy,
        'Cluster + above 200dma': base & cluster & f['above_200'],
        'Cluster + damaged (the Intel shape)':
            base & cluster & (f['dd_3y'] <= -0.40) & f['stale_peak'].fillna(False),
        'Cluster + heavy': base & cluster & heavy,
        'NO insider buying (90d)': base & ~any_buy,
        'BASELINE: every eligible name': base,
    }

    rets = close.pct_change(fill_method=None)
    ew = rets.where(base_full).mean(axis=1).fillna(0.0)
    ew_px = (1 + ew).cumprod()

    rows = []
    for label, mask in setups.items():
        m = mask.loc[start:]
        row = {'setup': label, 'signals': int(m.values.sum())}
        for h, hn in ((63, '3m'), (126, '6m'), (252, '12m')):
            fwd = ((close.shift(-h) / close - 1.0) * 100).loc[start:]
            b = ((ew_px.shift(-h) / ew_px - 1.0) * 100).loc[start:]
            r = fwd.where(m).stack().dropna()
            if len(r) < 200:
                continue
            ex = fwd.sub(b, axis=0).where(m).stack().dropna()
            row[f'{hn} med'] = r.median()
            row[f'{hn} vs peers'] = ex.median()
            row[f'{hn} beat peers %'] = (ex > 0).mean() * 100
        rows.append(row)
    tbl = pd.DataFrame(rows)

    n_elig = base_full.loc[start:].sum(axis=1)
    pct_buy = (base_full & any_buy).loc[start:].sum(axis=1) / n_elig
    print()
    print('=' * 165)
    print(f'{name.upper()}   {start.date()} -> {close.index[-1].date()}   '
          f'median {n_elig.median():.0f} eligible names, '
          f'{100 * pct_buy.median():.1f}% with insider buying in the trailing 90 days')
    print('=' * 165)
    print(tbl.round(2).to_string(index=False))

    # ------------------------------------------------------------------ portfolio
    rets0 = rets.fillna(0.0)
    ew_ret = ew
    rows = []
    for label in ['Cluster: 3+ insiders (90d)', 'Cluster: 5+ insiders (90d)',
                  'Cluster + above 200dma', 'Cluster + heavy',
                  'Cluster + damaged (the Intel shape)',
                  'Very heavy: >2 days of volume',
                  'NO insider buying (90d)', 'BASELINE: every eligible name']:
        m = setups[label]
        if m.loc[start:].values.sum() < 200:
            continue
        cs = [cagr(equity(m, close, rets0, seed=s).loc[start:]) for s in range(5)]
        rows.append({'strategy': label, 'CAGR mean': np.mean(cs),
                     'best draw': max(cs), 'worst draw': min(cs),
                     'luck spread': max(cs) - min(cs),
                     'vs peers': np.mean(cs) - cagr(ew_ret.loc[start:])})
    res = pd.DataFrame(rows).sort_values('vs peers', ascending=False)
    print()
    print(f'PORTFOLIO  --  20 equal-weight names, 6-month holds, 20bps, 5 random draws')
    print(f'  equal-weight universe CAGR: {cagr(ew_ret.loc[start:]):.2f}%')
    print(res.round(2).to_string(index=False))

    tag = cfg['member'].split('_')[0]
    tbl.to_csv(DATA / f'insider_signals_{tag}.csv', index=False)
    res.to_csv(DATA / f'insider_portfolio_{tag}.csv', index=False)
    return tbl, res


def main():
    d = load_prices()
    print('Assigning industries...', flush=True)
    industry = assign_industry(d['close'], d['etf'])
    print('Building features...', flush=True)
    f = build_features(d, industry)

    df = clean_filings(d['close'])
    print('\nBuilding trailing-90-day insider matrices...', flush=True)
    n_ins, usd = build_matrices(df, d['close'])

    for name, cfg in UNIVERSES.items():
        evaluate(name, cfg, d, f, n_ins, usd)
    print('\nSaved -> research/data/insider_*.csv')


if __name__ == '__main__':
    main()
