"""Does anything work outside large caps?

The S&P 500 study found no setup with a real edge. Market-efficiency arguments say
that is the expected result there: large caps are the most heavily researched
segment. Anomalies are supposed to survive in smaller, less-covered names.

Two universes, each restricted to the window its membership log actually covers:
  S&P MidCap 400   -- change log starts 2012, so 2013-01 onwards
  S&P SmallCap 600 -- change log starts 2019-12, so 2020-07 onwards

The benchmark for every test is an equal-weight index of the eligible universe
itself. That is the honest question: not "did it beat SPY" (a size and style bet)
but "did picking these names beat picking names at random from the same pool".

Run:  venv/bin/python3 research/smid_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA, SECTOR_ETFS, assign_industry

pd.set_option('display.width', 280)
pd.set_option('display.max_columns', 60)

TRADING_YEAR = 252

UNIVERSES = {
    'S&P 400 mid caps': dict(member='sp400_membership.parquet', bench='IJH',
                             start='2013-01-01', min_dv=10e6),
    'S&P 600 small caps': dict(member='sp600_membership.parquet', bench='IJR',
                               start='2020-07-01', min_dv=3e6),
}


# --------------------------------------------------------------------------- data

def load_prices():
    fields = {}
    for fld in ('close', 'open', 'high', 'low', 'volume'):
        a = pd.read_parquet(DATA / f'px_{fld}.parquet')
        b = pd.read_parquet(DATA / f'px_smid_{fld}.parquet')
        idx = a.index.union(b.index)
        a = a.reindex(idx)
        b = b.reindex(idx)
        keep = [c for c in b.columns if c not in a.columns]
        fields[fld] = pd.concat([a, b[keep]], axis=1).sort_index()

    close = fields['close'].copy()
    close[close <= 0.01] = np.nan
    rets = close.pct_change(fill_method=None)
    bad = (rets.abs() > 1.5)
    bad_cols = bad.any()[bad.any()].index.tolist()
    for c in bad_cols:
        close.loc[bad[c].idxmax():, c] = np.nan
    fields['close'] = close
    for k in ('open', 'high', 'low'):
        m = fields[k]
        m[m <= 0.01] = np.nan
        fields[k] = m.where(close.notna())
    print(f'Price matrix: {close.shape[0]} days x {close.shape[1]} tickers '
          f'({len(bad_cols)} corrupted series truncated)')

    etf = pd.read_parquet(DATA / 'etf_close.parquet')
    etf2 = pd.read_parquet(DATA / 'etf_smid_close.parquet')
    etf = etf.reindex(close.index).ffill()
    etf2 = etf2.reindex(close.index).ffill()
    for c in etf2.columns:
        if c not in etf.columns:
            etf[c] = etf2[c]
    fields['etf'] = etf
    return fields


def build_features(d, industry):
    close, etf, vol = d['close'], d['etf'], d['volume']
    f = {'close': close}

    peak3y = close.rolling(3 * TRADING_YEAR, min_periods=250).max()
    f['dd_3y'] = close / peak3y - 1.0
    # The 3y peak is "stale" when the recent 6-month high is strictly below it.
    f['stale_peak'] = close.rolling(126, min_periods=100).max() < peak3y * 0.999

    sma200 = close.rolling(200, min_periods=200).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    f['sma200'] = sma200
    f['above_200'] = close > sma200
    f['above_50'] = close > sma50
    f['sma200_slope'] = (sma200 / sma200.shift(21) - 1.0) * 100

    for lb, name in ((21, '1m'), (63, '3m'), (126, '6m'), (252, '12m')):
        f[f'ret_{name}'] = (close / close.shift(lb) - 1.0) * 100

    hi52 = close.rolling(252, min_periods=200).max()
    f['near_high'] = close >= hi52 * 0.95

    f['dollar_vol'] = (close * vol).rolling(60, min_periods=30).median()
    f['vol_ann'] = close.pct_change(fill_method=None).rolling(
        60, min_periods=40).std() * np.sqrt(252) * 100
    f['age_days'] = close.notna().cumsum()

    e = etf[SECTOR_ETFS]
    e_ret12 = (e / e.shift(252) - 1.0) * 100
    e_above = e > e.rolling(200, min_periods=200).mean()
    ind = industry.reindex(close.index)

    def map_ind(src):
        out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for name in SECTOR_ETFS:
            mask = (ind == name)
            if mask.values.any():
                out = out.mask(mask, src[name], axis=0)
        return out

    f['ind_ret_12m'] = map_ind(e_ret12)
    f['ind_above_200'] = map_ind(e_above.astype(float)) > 0.5
    f['rel_12m'] = f['ret_12m'] - f['ind_ret_12m']
    return f


# ------------------------------------------------------------------------ setups

def make_setups(f, base, close):
    above200 = f['above_200']
    reclaim = above200 & (~above200).rolling(63, min_periods=1).min().astype(bool).shift(1).fillna(False)
    damaged = (f['dd_3y'] <= -0.40) & f['stale_peak'].fillna(False)
    ind_up = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)

    mom = (close.shift(21) / close.shift(21 + 252) - 1.0).where(base)
    mom_rank = mom.rank(axis=1, pct=True)
    lowvol_rank = f['vol_ann'].where(base).rank(axis=1, pct=True)
    rev_rank = f['ret_1m'].where(base).rank(axis=1, pct=True)

    return {
        'Intel playbook (damaged + hot industry + turn)':
            base & damaged & ind_up & above200,
        'Damaged + 200dma reclaim':
            base & damaged & reclaim,
        'Deep damage (-60%) + reclaim':
            base & (f['dd_3y'] <= -0.60) & f['stale_peak'].fillna(False) & reclaim,
        'Momentum: top 20% 12m, above 200dma':
            base & (mom_rank >= 0.80) & above200,
        'Momentum: top 10% 12m, above 200dma':
            base & (mom_rank >= 0.90) & above200,
        'Low volatility: quietest 20%':
            base & (lowvol_rank <= 0.20),
        'Low vol + above 200dma':
            base & (lowvol_rank <= 0.20) & above200,
        'Short-term reversal: worst 10% over 1m':
            base & (rev_rank <= 0.10),
        'Reversal within an uptrend':
            base & (rev_rank <= 0.20) & above200 & (f['sma200_slope'] > 0),
        'Near 52-week high':
            base & f['near_high'].fillna(False),
        'BASELINE: every eligible name':
            base,
    }, mom


# ------------------------------------------------------------------ evaluation

def signal_table(masks, close, base, bench_px, start):
    """Median forward return vs an equal-weight index of the eligible universe."""
    rets = close.pct_change(fill_method=None)
    ew = rets.where(base).mean(axis=1).fillna(0.0)
    ew_px = (1 + ew).cumprod()

    rows = []
    for label, mask in masks.items():
        mask = mask.loc[start:]
        row = {'setup': label, 'signals': int(mask.values.sum())}
        for h, hn in ((63, '3m'), (126, '6m'), (252, '12m')):
            fwd = (close.shift(-h) / close - 1.0) * 100
            b_ew = (ew_px.shift(-h) / ew_px - 1.0) * 100
            b_ix = (bench_px.shift(-h) / bench_px - 1.0) * 100
            fwd = fwd.loc[start:]
            r = fwd.where(mask).stack().dropna()
            if len(r) < 200:
                continue
            ex = fwd.sub(b_ew.loc[start:], axis=0).where(mask).stack().dropna()
            ei = fwd.sub(b_ix.loc[start:], axis=0).where(mask).stack().dropna()
            row[f'{hn} med'] = r.median()
            row[f'{hn} vs peers'] = ex.median()
            row[f'{hn} beat peers %'] = (ex > 0).mean() * 100
            row[f'{hn} vs index'] = ei.median()
        rows.append(row)
    return pd.DataFrame(rows), ew


def equity(mask, close, rets, hold=126, max_pos=20, seed=0, cost_bps=20):
    idx = close.index
    entry_days = set(pd.Series(idx, index=idx).groupby(idx.to_period('M')).last())
    weights = pd.DataFrame(0.0, index=idx, columns=close.columns)
    openp = {}
    mv = mask.values
    cols = list(close.columns)
    cpos = {c: i for i, c in enumerate(cols)}
    rng = np.random.default_rng(seed)

    for i, dt in enumerate(idx):
        for t in [t for t, e in openp.items() if e <= i]:
            openp.pop(t)
        if dt in entry_days:
            lo = max(0, i - 21)
            recent = mv[lo:i + 1].any(axis=0)
            cand = [j for j in np.where(recent)[0] if cols[j] not in openp]
            room = max_pos - len(openp)
            if room > 0 and cand:
                for j in rng.permutation(cand)[:room]:
                    openp[cols[j]] = min(i + hold, len(idx) - 1)
        if openp:
            w = 1.0 / len(openp)
            for t in openp:
                weights.iat[i, cpos[t]] = w

    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    return port - dw * (cost_bps / 2 / 10000.0)


def cagr(r):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return (eq.iloc[-1] ** (1 / yrs) - 1) * 100


def run(name, cfg, d, f, industry):
    close = d['close']
    start = pd.Timestamp(cfg['start'])
    member = pd.read_parquet(DATA / cfg['member'])
    member = member.reindex(close.index.union(member.index)).ffill()
    member = member.reindex(close.index).fillna(False).astype(bool)
    member = member.reindex(columns=close.columns).fillna(False)

    liquid = (f['dollar_vol'] >= cfg['min_dv']) & (close >= 5.0)
    weekly = pd.DataFrame(np.repeat((close.index.dayofweek == 4)[:, None],
                                    close.shape[1], axis=1),
                          index=close.index, columns=close.columns)
    base_full = member & liquid & close.notna() & (f['age_days'] >= 400)
    base = base_full & weekly

    n_elig = base_full.loc[start:].sum(axis=1)
    print()
    print('=' * 165)
    print(f'{name.upper()}   window {start.date()} -> {close.index[-1].date()}   '
          f'eligible names per day: median {n_elig.median():.0f}, '
          f'min {n_elig.min():.0f}, max {n_elig.max():.0f}')
    print('=' * 165)

    masks, _ = make_setups(f, base, close)
    bench_px = d['etf'][cfg['bench']]
    tbl, ew = signal_table(masks, close, base_full, bench_px, start)
    print(tbl.round(2).to_string(index=False))
    print('\n"vs peers" = median forward return minus an equal-weight index of the same')
    print('eligible universe. That column is the test: it is what picking at random gives.')

    # ------------------------------------------------------ portfolio, with luck bands
    rets = close.pct_change(fill_method=None).fillna(0.0)
    ew_ret = close.pct_change(fill_method=None).where(base_full).mean(axis=1).fillna(0.0)
    bench_ret = bench_px.pct_change(fill_method=None).fillna(0.0)

    interesting = [k for k in masks if not k.startswith('BASELINE')]
    rows = []
    for label in interesting + ['BASELINE: every eligible name']:
        m = masks[label]
        if m.loc[start:].values.sum() < 200:
            continue
        cs = []
        for seed in range(5):
            p = equity(m, close, rets, seed=seed).loc[start:]
            cs.append(cagr(p))
        ewc = cagr(ew_ret.loc[start:])
        bc = cagr(bench_ret.loc[start:])
        rows.append({'strategy': label, 'CAGR mean': np.mean(cs),
                     'best draw': max(cs), 'worst draw': min(cs),
                     'luck spread': max(cs) - min(cs),
                     'vs peers': np.mean(cs) - ewc, 'vs index': np.mean(cs) - bc})
    res = pd.DataFrame(rows).sort_values('vs peers', ascending=False)
    print()
    print(f'PORTFOLIO  --  20 equal-weight names, 6-month holds, 20bps, 5 random draws each')
    print(f'  equal-weight universe CAGR: {cagr(ew_ret.loc[start:]):.2f}%     '
          f'{cfg["bench"]} CAGR: {cagr(bench_ret.loc[start:]):.2f}%')
    print(res.round(2).to_string(index=False))
    tbl.to_csv(DATA / f'smid_signals_{cfg["bench"]}.csv', index=False)
    res.to_csv(DATA / f'smid_portfolio_{cfg["bench"]}.csv', index=False)
    return tbl, res


def main():
    d = load_prices()
    print('Assigning industries by trailing return correlation...', flush=True)
    industry = assign_industry(d['close'], d['etf'])
    print('Building features...', flush=True)
    f = build_features(d, industry)
    for name, cfg in UNIVERSES.items():
        run(name, cfg, d, f, industry)
    print('\nSaved -> research/data/smid_*.csv')


if __name__ == '__main__':
    main()
