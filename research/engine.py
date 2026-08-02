"""Feature engine + event-study backtest for the Laggard playbook.

Everything is computed from a (date x ticker) price matrix using trailing windows
only. Entries fill at the next session's open. Industry membership is inferred from
trailing return correlation against sector ETFs rather than from yfinance's current
sector labels, which would leak today's classification into the past.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / 'data'

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
               'SMH', 'IBB', 'KRE', 'XRT', 'OIH', 'ITB']

TRADING_YEAR = 252


# ----------------------------------------------------------------------------- data

def load():
    close = pd.read_parquet(DATA / 'px_close.parquet').sort_index()
    open_ = pd.read_parquet(DATA / 'px_open.parquet').sort_index()
    high = pd.read_parquet(DATA / 'px_high.parquet').sort_index()
    low = pd.read_parquet(DATA / 'px_low.parquet').sort_index()
    vol = pd.read_parquet(DATA / 'px_volume.parquet').sort_index()
    etf = pd.read_parquet(DATA / 'etf_close.parquet').sort_index()
    member = pd.read_parquet(DATA / 'sp500_membership.parquet')

    idx = close.index
    etf = etf.reindex(idx).ffill()
    # Membership is monthly; forward-fill onto trading days, shifted so that a
    # month-end snapshot only becomes usable the following day.
    member = member.reindex(idx.union(member.index)).ffill().reindex(idx).fillna(False)
    return dict(close=close, open=open_, high=high, low=low, volume=vol,
                etf=etf, member=member.astype(bool))


# ------------------------------------------------------------------------- features

def assign_industry(close, etf, window=250, step=21):
    """Map each stock to the sector ETF its returns track most closely.

    Recomputed every `step` days on a trailing `window`, so the label at time t
    only reflects information available at t.
    """
    rets = close.pct_change(fill_method=None)
    erets = etf[SECTOR_ETFS].pct_change(fill_method=None)

    dates = close.index[::step]
    labels = pd.DataFrame(index=close.index, columns=close.columns, dtype=object)

    for d in dates:
        lo = close.index.get_loc(d)
        if lo < window:
            continue
        r = rets.iloc[lo - window:lo]
        e = erets.iloc[lo - window:lo]
        valid = r.columns[r.notna().sum() >= window * 0.6]
        if not len(valid):
            continue
        r = r[valid]
        # correlation of each stock against each ETF
        rz = (r - r.mean()) / r.std(ddof=0)
        ez = (e - e.mean()) / e.std(ddof=0)
        rz = rz.fillna(0.0)
        ez = ez.fillna(0.0)
        corr = pd.DataFrame(rz.values.T @ ez.values / len(r),
                            index=valid, columns=e.columns)
        best = corr.idxmax(axis=1)
        labels.loc[d, valid] = best.values

    labels = labels.ffill()
    return labels


def build_features(d, industry):
    close, etf = d['close'], d['etf']
    vol = d['volume']

    f = {}
    f['close'] = close

    peak3y = close.rolling(3 * TRADING_YEAR, min_periods=250).max()
    f['dd_3y'] = close / peak3y - 1.0

    # How stale is the old high? A fresh peak means the stock is not "beaten down",
    # it just had a pullback.
    argmax = close.rolling(3 * TRADING_YEAR, min_periods=250).apply(
        lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    f['peak_age_days'] = argmax

    sma200 = close.rolling(200, min_periods=200).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    f['sma200'] = sma200
    f['above_200'] = close > sma200
    f['dist_200_pct'] = (close / sma200 - 1.0) * 100
    f['sma200_slope'] = (sma200 / sma200.shift(21) - 1.0) * 100
    f['above_50'] = close > sma50

    for lb, name in ((21, '1m'), (63, '3m'), (126, '6m'), (252, '12m')):
        f[f'ret_{name}'] = (close / close.shift(lb) - 1.0) * 100

    low52 = close.rolling(252, min_periods=200).min()
    f['off_52w_low_pct'] = (close / low52 - 1.0) * 100

    dollar_vol = (close * vol).rolling(60, min_periods=30).median()
    f['dollar_vol'] = dollar_vol

    rets = close.pct_change(fill_method=None)
    f['vol_ann'] = rets.rolling(60, min_periods=40).std() * np.sqrt(252) * 100

    f['age_days'] = close.notna().cumsum()

    # --- industry aggregates, mapped back onto each stock
    e = etf[SECTOR_ETFS]
    e_ret12 = (e / e.shift(252) - 1.0) * 100
    e_sma200 = e.rolling(200, min_periods=200).mean()
    e_above = e > e_sma200
    e_ret3 = (e / e.shift(63) - 1.0) * 100

    ind = industry.reindex(close.index)

    def map_ind(src):
        out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for etf_name in SECTOR_ETFS:
            mask = (ind == etf_name)
            if not mask.values.any():
                continue
            col = src[etf_name]
            out = out.mask(mask, col, axis=0)
        return out

    f['ind_ret_12m'] = map_ind(e_ret12)
    f['ind_ret_3m'] = map_ind(e_ret3)
    f['ind_above_200'] = map_ind(e_above.astype(float)) > 0.5
    f['industry'] = ind

    f['rel_12m'] = f['ret_12m'] - f['ind_ret_12m']
    f['rel_3m'] = f['ret_3m'] - f['ind_ret_3m']

    # market backdrop
    spy = etf['SPY']
    f['spy_above_200'] = pd.DataFrame(
        np.repeat((spy > spy.rolling(200, min_periods=200).mean()).values[:, None],
                  close.shape[1], axis=1),
        index=close.index, columns=close.columns)

    return f


# ------------------------------------------------------------------------- signals

def make_signal(f, d, cfg):
    """Boolean (date x ticker) matrix of entry candidates."""
    close = f['close']
    member = d['member'].reindex(columns=close.columns).fillna(False)

    liquid = (f['dollar_vol'] >= cfg['min_dollar_vol']) & (close >= cfg['min_price'])
    seasoned = f['age_days'] >= cfg['min_age_days']

    damage = f['dd_3y'] <= cfg['max_dd']
    stale_peak = f['peak_age_days'] >= cfg['min_peak_age']

    cond = member & liquid & seasoned & damage & stale_peak

    if cfg.get('require_industry_up', True):
        cond &= f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= cfg['min_ind_ret'])

    if cfg.get('require_divergence', True):
        cond &= f['rel_12m'] <= cfg['max_rel_12m']

    if cfg.get('max_vol') is not None:
        cond &= f['vol_ann'] <= cfg['max_vol']

    if cfg.get('require_market_up', False):
        cond &= f['spy_above_200'].fillna(False)

    trigger = cfg.get('trigger', 'reclaim_200')
    if trigger == 'none':
        trig = pd.DataFrame(True, index=close.index, columns=close.columns)
    elif trigger == 'above_200':
        trig = f['above_200']
    elif trigger == 'reclaim_200':
        above = f['above_200']
        # first close above the 200dma after being below it for a sustained stretch
        below_run = (~above).rolling(cfg['reclaim_lookback'], min_periods=1).min().astype(bool)
        trig = above & below_run.shift(1).fillna(False)
    elif trigger == 'reclaim_50':
        above = f['above_50']
        below_run = (~above).rolling(cfg['reclaim_lookback'], min_periods=1).min().astype(bool)
        trig = above & below_run.shift(1).fillna(False)
    elif trigger == 'breakout_3m':
        hi3m = close.rolling(63, min_periods=40).max()
        trig = close >= hi3m
    elif trigger == 'higher_low':
        # 3m low is above the 6m low -> the downtrend has stopped making new lows
        lo3 = close.rolling(63, min_periods=40).min()
        lo6 = close.rolling(126, min_periods=80).min()
        trig = (lo3 > lo6 * 1.02) & f['above_50']
    else:
        raise ValueError(trigger)

    sig = cond & trig.fillna(False)

    # Debounce: one entry per ticker per cooldown window.
    if cfg.get('cooldown', 0):
        sig = _debounce(sig, cfg['cooldown'])
    return sig


def _debounce(sig, cooldown):
    arr = sig.values.copy()
    n, m = arr.shape
    last = np.full(m, -10 ** 9)
    for i in range(n):
        row = arr[i]
        hit = np.where(row)[0]
        for j in hit:
            if i - last[j] < cooldown:
                arr[i, j] = False
            else:
                last[j] = i
    return pd.DataFrame(arr, index=sig.index, columns=sig.columns)


# ------------------------------------------------------------------------ simulate

def simulate(sig, f, d, cfg):
    """Event study: fill at next open, then apply the exit policy."""
    close, open_, low, high = d['close'], d['open'], d['low'], d['high']
    spy = d['etf']['SPY']
    idx = close.index
    pos = {t: i for i, t in enumerate(close.columns)}

    C = close.values
    O = open_.reindex(index=idx, columns=close.columns).values
    L = low.reindex(index=idx, columns=close.columns).values
    H = high.reindex(index=idx, columns=close.columns).values
    S = spy.reindex(idx).values
    SMA = f['sma200'].values

    max_hold = cfg['max_hold']
    stop = cfg.get('stop_pct')
    target = cfg.get('target_pct')
    trail = cfg.get('trail_pct')
    thesis_break = cfg.get('thesis_break_days')
    slippage = cfg.get('slippage_bps', 10) / 10000.0

    rows = []
    sig_idx = np.where(sig.values)
    for i, j in zip(*sig_idx):
        entry_i = i + 1
        if entry_i + 5 >= len(idx):
            continue
        ep = O[entry_i, j]
        if not np.isfinite(ep) or ep <= 0:
            continue
        ep *= (1 + slippage)

        end = min(entry_i + max_hold, len(idx) - 1)
        exit_i, exit_p, reason = end, C[end, j], 'time'

        peak = ep
        below_count = 0
        for k in range(entry_i, end + 1):
            lo_k, hi_k, cl_k = L[k, j], H[k, j], C[k, j]
            if not np.isfinite(cl_k):
                continue
            if stop is not None and np.isfinite(lo_k) and lo_k <= ep * (1 - stop):
                exit_i, exit_p, reason = k, ep * (1 - stop), 'stop'
                break
            if target is not None and np.isfinite(hi_k) and hi_k >= ep * (1 + target):
                exit_i, exit_p, reason = k, ep * (1 + target), 'target'
                break
            peak = max(peak, hi_k if np.isfinite(hi_k) else cl_k)
            if trail is not None and cl_k <= peak * (1 - trail):
                exit_i, exit_p, reason = k, cl_k, 'trail'
                break
            if thesis_break is not None and np.isfinite(SMA[k, j]):
                below_count = below_count + 1 if cl_k < SMA[k, j] else 0
                if below_count >= thesis_break:
                    exit_i, exit_p, reason = k, cl_k, 'thesis_break'
                    break

        exit_p *= (1 - slippage)
        ret = exit_p / ep - 1.0

        seg_lo = np.nanmin(L[entry_i:exit_i + 1, j]) if exit_i >= entry_i else ep
        mae = seg_lo / ep - 1.0
        seg_hi = np.nanmax(H[entry_i:exit_i + 1, j]) if exit_i >= entry_i else ep
        mfe = seg_hi / ep - 1.0

        spy_ret = S[exit_i] / S[entry_i] - 1.0 if np.isfinite(S[entry_i]) else np.nan

        rows.append(dict(
            signal_date=idx[i], entry_date=idx[entry_i], exit_date=idx[exit_i],
            ticker=close.columns[j], entry=ep, exit=exit_p,
            ret=ret * 100, spy_ret=spy_ret * 100, excess=(ret - spy_ret) * 100,
            hold_days=exit_i - entry_i, reason=reason,
            mae=mae * 100, mfe=mfe * 100,
            dd_3y=f['dd_3y'].values[i, j] * 100,
            rel_12m=f['rel_12m'].values[i, j],
            ret_12m=f['ret_12m'].values[i, j],
            ind_ret_12m=f['ind_ret_12m'].values[i, j],
            industry=f['industry'].values[i, j],
            vol_ann=f['vol_ann'].values[i, j],
        ))

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- report

def summarize(trades, label=''):
    if trades.empty:
        return {'label': label, 'n': 0}
    r, x = trades['ret'], trades['excess']
    return {
        'label': label,
        'n': len(trades),
        'n_names': trades['ticker'].nunique(),
        'win_%': (r > 0).mean() * 100,
        'mean_%': r.mean(),
        'median_%': r.median(),
        'beat_spy_%': (x > 0).mean() * 100,
        'mean_excess_%': x.mean(),
        'median_excess_%': x.median(),
        'p10_%': r.quantile(0.10),
        'p90_%': r.quantile(0.90),
        'mean_mae_%': trades['mae'].mean(),
        'worst_%': r.min(),
        'avg_hold': trades['hold_days'].mean(),
    }


DEFAULT_CFG = dict(
    min_dollar_vol=20e6,
    min_price=5.0,
    min_age_days=400,
    max_dd=-0.40,
    min_peak_age=126,
    require_industry_up=True,
    min_ind_ret=0.0,
    require_divergence=True,
    max_rel_12m=-25.0,
    max_vol=None,
    require_market_up=False,
    trigger='reclaim_200',
    reclaim_lookback=63,
    cooldown=126,
    max_hold=126,
    stop_pct=None,
    target_pct=None,
    trail_pct=None,
    thesis_break_days=None,
    slippage_bps=10,
)
