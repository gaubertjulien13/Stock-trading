"""
Hardened, multi-period backtest harness for the buy-signal scoring system.

This replaces an earlier single-month backtest and fixes its lookahead bias. The key
correctness change: daily trend context (SMA200, ADX, ATR, relative strength,
and the market regime) is evaluated *as of each trade's own date* using only
the most recent COMPLETED daily bar (day D-1) — never end-of-window data.

What it does, per period window:
  1. Reuse the live script's intraday indicators + scoring untouched.
  2. Score every bar with the daily context that was actually knowable then.
  3. On a fresh threshold cross, enter at the NEXT bar's open (no same-bar peek).
  4. Resolve the trade against 1.5x-ATR stop / 3x-ATR target over a hold horizon
     expressed in trading days (your 1-5 day swing horizon).
  5. Report expectancy-first metrics per window, plus train/test aggregates and
     cross-window stability (to catch regime-fragile / overfit configurations).

Data reality (Yahoo/yfinance): 15m bars only go back ~60 trading days (one
regime), while 1h bars go back ~2 years (many regimes). So 1h is the primary
multi-regime interval; 15m is a recent confirmation run.

Usage:
    python backtest_harness.py                 # 1h, multi-window, liquid subset
    python backtest_harness.py --interval 15m  # recent 15m confirmation window
    python backtest_harness.py --refresh       # ignore on-disk data cache
    python backtest_harness.py --subset-size 60 --threshold 9 --hold-days 5
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import sys
import pickle
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np

from myutils import get_sp500_tickers
from Trading_Buy_Signal_Strict_Script_With_Alerts import (
    _chunks, _extract_cols, _slice_from_batch, _compute_adx,
    compute_signal_score,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# Approx number of regular-session bars per trading day, by interval.
BARS_PER_DAY = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '60m': 7}

# yfinance max intraday lookback (period string) by interval.
MAX_PERIOD = {'1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d', '1h': '730d', '60m': '730d'}


# =========================================================================
# Period windows (train = fit later steps here; test = untouched holdout)
# =========================================================================
# 1h history reaches ~2 years back, so we sample non-overlapping ~2-month
# windows across different market conditions. The most recent window is the
# holdout. SPY return per window is printed at runtime so the regime is visible
# empirically rather than assumed by a hand-typed label.
WINDOWS_1H = [
    {'name': '2024-Q1', 'start': '2024-02-01', 'end': '2024-04-01', 'split': 'train'},
    {'name': '2024-Summer', 'start': '2024-06-01', 'end': '2024-08-01', 'split': 'train'},
    {'name': '2024-Q4', 'start': '2024-10-01', 'end': '2024-12-01', 'split': 'train'},
    {'name': '2025-Q1', 'start': '2025-02-01', 'end': '2025-04-01', 'split': 'train'},
    {'name': '2025-Summer', 'start': '2025-07-01', 'end': '2025-09-01', 'split': 'train'},
    {'name': '2026-Q1', 'start': '2026-01-01', 'end': '2026-03-01', 'split': 'test'},
]

# 15m only has ~60 trading days; one recent window for confirmation.
def _recent_15m_windows():
    end = datetime.today()
    start = end - timedelta(days=55)
    return [{
        'name': 'recent-15m',
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'split': 'test',
    }]


# =========================================================================
# Data download + on-disk cache
# =========================================================================

def _cache_path(kind, interval):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{kind}_{interval}.pkl")


def _download_batched(tickers, batch_size=20, **dl_kwargs):
    out = {}
    for batch in _chunks(tickers, batch_size):
        try:
            data = yf.download(batch, progress=False, group_by='ticker',
                               threads=True, **dl_kwargs)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        sub = _slice_from_batch(data, tkr)
                        if sub is not None and not sub.empty:
                            out[tkr] = sub
                    except Exception:
                        pass
            elif len(batch) == 1:
                out[batch[0]] = data.dropna().copy()
        except Exception as e:
            print(f"  download error: {str(e)[:80]}")
    return out


def load_daily(tickers, refresh=False):
    path = _cache_path("daily", "1d")
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if set(tickers).issubset(cached.keys()) or len(cached) >= len(tickers):
            print(f"  using cached daily data ({len(cached)} tickers)")
            return cached
    print("  downloading daily bars (trailing ~3 years)...")
    end = datetime.today()
    start = end - timedelta(days=3 * 365 + 250)  # extra for SMA200 warmup
    daily = _download_batched(tickers, start=start, end=end)
    with open(path, "wb") as f:
        pickle.dump(daily, f)
    print(f"  cached daily data for {len(daily)} tickers")
    return daily


def load_intraday(tickers, interval, refresh=False):
    path = _cache_path("intraday", interval)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if set(tickers).issubset(cached.keys()):
            print(f"  using cached {interval} data ({len(cached)} tickers)")
            return cached
    period = MAX_PERIOD.get(interval, '60d')
    print(f"  downloading {interval} bars (period={period})...")
    intra = _download_batched(tickers, period=period, interval=interval)
    with open(path, "wb") as f:
        pickle.dump(intra, f)
    print(f"  cached {interval} data for {len(intra)} tickers")
    return intra


def load_earnings(tickers, refresh=False):
    """Per-ticker set of historical+future earnings dates (yfinance). Cached."""
    path = _cache_path("earnings", "dates")
    cached = {}
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if set(tickers).issubset(cached.keys()):
            print(f"  using cached earnings dates ({len(cached)} tickers)")
            return cached
    print("  fetching earnings dates...")
    for tkr in tickers:
        if tkr in cached:
            continue
        try:
            ed = yf.Ticker(tkr).get_earnings_dates(limit=24)
            if ed is not None and len(ed):
                idx = ed.index
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                cached[tkr] = set(pd.to_datetime(idx).normalize().date)
            else:
                cached[tkr] = set()
        except Exception:
            cached[tkr] = set()
    with open(path, "wb") as f:
        pickle.dump(cached, f)
    print(f"  cached earnings dates for {len(cached)} tickers")
    return cached


# GICS (constituents CSV) and yfinance sector names -> SPDR sector ETF proxy.
SECTOR_ETF = {
    'Information Technology': 'XLK', 'Technology': 'XLK',
    'Financials': 'XLF', 'Financial Services': 'XLF',
    'Health Care': 'XLV', 'Healthcare': 'XLV',
    'Consumer Discretionary': 'XLY', 'Consumer Cyclical': 'XLY',
    'Consumer Staples': 'XLP', 'Consumer Defensive': 'XLP',
    'Energy': 'XLE', 'Industrials': 'XLI',
    'Materials': 'XLB', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE', 'Utilities': 'XLU',
    'Communication Services': 'XLC', 'Telecommunications': 'XLC',
}


def load_sector_map(refresh=False):
    """{ticker: GICS sector} from the same constituents CSV used for tickers."""
    path = _cache_path("sectors", "map")
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    import requests
    from io import StringIO
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(StringIO(requests.get(url, timeout=20).text))
    m = {str(s).strip().replace('.', '-').upper(): sec
         for s, sec in zip(df['Symbol'], df['GICS Sector'])}
    with open(path, "wb") as f:
        pickle.dump(m, f)
    return m


class SectorProvider:
    """As-of 5-day returns for the 11 SPDR sector ETFs (lookahead-safe)."""

    def __init__(self, refresh=False):
        etfs = sorted(set(SECTOR_ETF.values()))
        path = _cache_path("daily", "sector_etfs")
        if not refresh and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            end = datetime.today()
            start = end - timedelta(days=3 * 365 + 60)
            data = _download_batched(etfs, start=start, end=end)
            with open(path, "wb") as f:
                pickle.dump(data, f)
        self.ret5 = {}
        self.dates = {}
        for etf, df in data.items():
            try:
                close, _, _, _ = _extract_cols(df, ticker=etf)
                idx = close.index
                idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                self.ret5[etf] = (close.pct_change(5) * 100.0).to_numpy()
                self.dates[etf] = idx.values.astype('datetime64[D]')
            except Exception:
                continue

    def get(self, sector_name, trade_date):
        """Sector ETF 5d return %, as of the last completed daily bar before trade_date."""
        etf = SECTOR_ETF.get(sector_name)
        if etf is None or etf not in self.ret5:
            return np.nan
        d = np.datetime64(pd.Timestamp(trade_date).normalize(), 'D')
        pos = np.searchsorted(self.dates[etf], d, side='left') - 1
        if pos < 0:
            return np.nan
        v = self.ret5[etf][pos]
        return float(v) if np.isfinite(v) else np.nan


def load_spy_daily(refresh=False):
    path = _cache_path("daily", "SPY")
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    end = datetime.today()
    start = end - timedelta(days=3 * 365 + 250)
    raw = yf.download("SPY", start=start, end=end, progress=False)
    close = None
    if raw is not None and not raw.empty:
        close, _, _, _ = _extract_cols(raw, ticker="SPY")
        close = close.astype(float)
    with open(path, "wb") as f:
        pickle.dump(close, f)
    return close


# =========================================================================
# Liquid subset selection (by average dollar volume)
# =========================================================================

def pick_liquid_subset(daily_data, n=80, lookback=60):
    scored = []
    for tkr, df in daily_data.items():
        try:
            close, _, _, vol = _extract_cols(df, ticker=tkr)
            if len(close) < lookback:
                continue
            dollar_vol = float((close.tail(lookback) * vol.tail(lookback)).mean())
            if np.isfinite(dollar_vol):
                scored.append((tkr, dollar_vol))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:n]]


# =========================================================================
# As-of daily context (the lookahead fix)
# =========================================================================

class DailyContextProvider:
    """
    Per-ticker daily context aligned to daily dates. A query for trade date D
    returns the context from the most recent COMPLETED daily bar (strictly
    before D), so intraday trades never see same-day or future daily data.
    """

    def __init__(self, daily_data, spy_close):
        self.ctx = {}           # tkr -> DataFrame indexed by date (cols below)
        self.dates = {}         # tkr -> np.datetime64[D] array of those dates
        self._build(daily_data, spy_close)
        self._build_regime(spy_close)

    def _build(self, daily_data, spy_close):
        spy_ret20 = None
        if spy_close is not None and len(spy_close) > 20:
            spy_idx = spy_close.index.normalize().tz_localize(None) if spy_close.index.tz is not None else spy_close.index.normalize()
            spy_ret20 = pd.Series(spy_close.pct_change(20).values, index=spy_idx)

        above_frames = {}  # for breadth

        for tkr, df in daily_data.items():
            try:
                close, high, low, vol = _extract_cols(df, ticker=tkr)
                if len(close) < 220:
                    continue
                idx = close.index
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                idx = idx.normalize()

                sma200 = close.rolling(200, min_periods=200).mean()
                above = (close > sma200)
                adx_series, atr_series = _compute_adx(high, low, close, period=14)
                ret20 = close.pct_change(20)

                rs_out = pd.Series(False, index=close.index)
                if spy_ret20 is not None:
                    tkr_ret20 = pd.Series(ret20.values, index=idx)
                    aligned_spy = spy_ret20.reindex(idx)
                    rs_vals = (tkr_ret20 > aligned_spy) & tkr_ret20.notna() & aligned_spy.notna()
                    rs_out = pd.Series(rs_vals.values, index=close.index)

                sma20 = close.rolling(20, min_periods=20).mean()
                ret5 = close.pct_change(5) * 100.0

                frame = pd.DataFrame({
                    'above_sma200': above.values,
                    'sma200': sma200.values,
                    'daily_close': close.values,
                    'adx': adx_series.values,
                    'daily_atr': atr_series.values,
                    'rs_outperforming': rs_out.values,
                    'above_sma20': (close > sma20).values,
                    'ret_5d': ret5.values,
                }, index=idx)
                frame['adx_trending'] = frame['adx'] > 25
                frame['adx_strong'] = frame['adx'] > 35

                self.ctx[tkr] = frame
                self.dates[tkr] = idx.values.astype('datetime64[D]')
                above_frames[tkr] = frame['above_sma200']
            except Exception:
                continue

        # Breadth: % of tickers above their SMA200 on each daily date.
        if above_frames:
            breadth_df = pd.DataFrame(above_frames)
            self.breadth = breadth_df.mean(axis=1, skipna=True) * 100.0
        else:
            self.breadth = pd.Series(dtype=float)

    def _build_regime(self, spy_close):
        """Per-date regime + threshold_adjust, mirroring compute_market_regime."""
        self.regime_dates = None
        self.regime_adjust = None
        self.regime_label = None
        if spy_close is None or len(spy_close) < 200:
            return
        idx = spy_close.index
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        close = pd.Series(spy_close.values, index=idx)
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ret5 = close.pct_change(5) * 100.0
        breadth = self.breadth.reindex(idx) if len(self.breadth) else pd.Series(100.0, index=idx)

        adjust = np.zeros(len(idx), dtype=int)
        label = np.array(['risk_on'] * len(idx), dtype=object)
        above50 = (close > sma50).values
        above200 = (close > sma200).values
        r5 = ret5.values
        br = breadth.values

        for i in range(len(idx)):
            if (not above200[i]) or (np.isfinite(r5[i]) and r5[i] < -5.0):
                label[i] = 'risk_off'
                adjust[i] = 99
            elif (not above50[i]) or (np.isfinite(br[i]) and br[i] < 40.0) or (np.isfinite(r5[i]) and r5[i] < -2.0):
                label[i] = 'caution'
                adjust[i] = 3
            else:
                label[i] = 'risk_on'
                adjust[i] = 0

        self.regime_dates = idx.values.astype('datetime64[D]')
        self.regime_adjust = adjust
        self.regime_label = label

    def get(self, tkr, trade_date):
        """Context dict for tkr as of the last completed daily bar before trade_date."""
        if tkr not in self.ctx:
            return None
        d = np.datetime64(pd.Timestamp(trade_date).normalize(), 'D')
        dates = self.dates[tkr]
        pos = np.searchsorted(dates, d, side='left') - 1
        if pos < 0:
            return None
        row = self.ctx[tkr].iloc[pos]
        if not np.isfinite(row['sma200']):
            return None
        return {
            'above_sma200': bool(row['above_sma200']),
            'sma200': float(row['sma200']),
            'daily_close': float(row['daily_close']),
            'adx': float(row['adx']) if np.isfinite(row['adx']) else 0.0,
            'adx_trending': bool(row['adx_trending']),
            'adx_strong': bool(row['adx_strong']),
            'daily_atr': float(row['daily_atr']) if np.isfinite(row['daily_atr']) else np.nan,
            'rs_outperforming': bool(row['rs_outperforming']),
            'above_sma20': bool(row['above_sma20']),
            'ret_5d': float(row['ret_5d']) if np.isfinite(row['ret_5d']) else 0.0,
        }

    def regime(self, trade_date):
        """(label, threshold_adjust) as of the last completed daily bar before trade_date."""
        if self.regime_dates is None:
            return 'risk_on', 0
        d = np.datetime64(pd.Timestamp(trade_date).normalize(), 'D')
        pos = np.searchsorted(self.regime_dates, d, side='left') - 1
        if pos < 0:
            return 'risk_on', 0
        return str(self.regime_label[pos]), int(self.regime_adjust[pos])


# =========================================================================
# Intraday indicators (identical math to the live script, on a full frame)
# =========================================================================

def compute_intraday_indicators_df(df, ticker=None):
    if df is None or df.empty or len(df) < 40:
        return None
    close, high, low, volume = _extract_cols(df, ticker=ticker)
    df = df.copy()
    df["Open"] = df["Open"].astype(float) if "Open" in df.columns else close
    df["Close"] = close
    df["High"] = high
    df["Low"] = low

    df["EMA5"] = close.ewm(span=5, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI_Rising"] = df["RSI"] > df["RSI"].shift(3)

    df["BB_Middle"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]).abs()
    df["BB_Range"] = df["BB_Width"].clip(lower=1e-9)
    bb_lower_bounce = (
        (close.shift(1) <= df["BB_Lower"].shift(1) * 1.02) &
        (close > df["BB_Lower"] * 1.02)
    )
    bb_middle_cross = (
        (close > df["BB_Middle"]) &
        (close.shift(1) <= df["BB_Middle"].shift(1))
    )
    bb_width_avg = df["BB_Width"].rolling(20, min_periods=20).mean()
    bb_width_ok = df["BB_Width"] > bb_width_avg * 0.8
    df["BB_Buy"] = (bb_lower_bounce | bb_middle_cross) & bb_width_ok
    df["BB_Lower_Bounce"] = bb_lower_bounce & bb_width_ok

    df["Volume_MA"] = volume.rolling(20, min_periods=20).mean()
    df["Volume_Above_Avg"] = volume > df["Volume_MA"]
    df["Volume_Surge"] = volume > df["Volume_MA"] * 1.5

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
    df["MACD_Hist_Positive"] = df["MACD_Hist"] > 0
    df["MACD_Hist_Rising"] = df["MACD_Hist"] > df["MACD_Hist"].shift(1)
    df["BB_Position"] = ((close - df["BB_Lower"]) / df["BB_Range"]) * 100.0

    df = df.dropna(subset=["RSI", "EMA5", "EMA20", "BB_Middle"])
    return df if not df.empty else None


def build_frames(intra_data, interval):
    """Precompute per-ticker indicator frames + raw price arrays (once)."""
    frames = {}
    arrays = {}
    for tkr, raw in intra_data.items():
        idf = compute_intraday_indicators_df(raw, ticker=tkr)
        if idf is None or idf.empty:
            continue
        frames[tkr] = idf
        arrays[tkr] = {
            'open': idf['Open'].to_numpy(dtype=float),
            'high': idf['High'].to_numpy(dtype=float),
            'low': idf['Low'].to_numpy(dtype=float),
            'close': idf['Close'].to_numpy(dtype=float),
            'index': idf.index,
        }
    return frames, arrays


# =========================================================================
# Exit engines  (each returns: result, exit_price, bars_held, risk_unit)
# risk_unit is the initial 1R distance, so r_multiple is comparable across exits.
# Same-bar stop+target is resolved conservatively as a loss.
# =========================================================================

def resolve_fixed(H, L, C, entry_idx, entry, atr, hold, stop_mult, target_mult):
    risk = stop_mult * atr
    stop = entry - risk
    target = entry + target_mult * atr
    n = len(C)
    for off in range(0, hold + 1):
        i = entry_idx + off
        if i >= n:
            break
        hit_stop = L[i] <= stop
        hit_target = H[i] >= target
        if hit_stop and hit_target:
            return 'loss', stop, off, risk
        if hit_target:
            return 'win', target, off, risk
        if hit_stop:
            return 'loss', stop, off, risk
    last = min(entry_idx + hold, n - 1)
    return 'timeout', C[last], last - entry_idx, risk


def resolve_trail(H, L, C, entry_idx, entry, atr, hold, trail_mult, init_stop_mult=1.5):
    risk = init_stop_mult * atr
    stop = entry - risk
    peak = H[entry_idx]
    n = len(C)
    for off in range(0, hold + 1):
        i = entry_idx + off
        if i >= n:
            break
        peak = max(peak, H[i])
        stop = max(stop, peak - trail_mult * atr)
        if L[i] <= stop:
            res = 'win' if stop > entry else 'loss'
            return res, stop, off, risk
    last = min(entry_idx + hold, n - 1)
    return 'timeout', C[last], last - entry_idx, risk


def resolve_scaleout(H, L, C, entry_idx, entry, atr, hold, trail_mult=2.0, partial_mult=1.5):
    """Take half at +partial_mult*ATR, move stop to breakeven, trail the rest."""
    risk = 1.5 * atr
    stop = entry - risk
    partial = entry + partial_mult * atr
    half_r = None  # locked R on the first half
    peak = H[entry_idx]
    n = len(C)
    for off in range(0, hold + 1):
        i = entry_idx + off
        if i >= n:
            break
        peak = max(peak, H[i])
        if half_r is None:
            if L[i] <= stop and H[i] >= partial:
                return 'loss', stop, off, risk
            if H[i] >= partial:
                half_r = (partial - entry) / risk
                continue
            if L[i] <= stop:
                return 'loss', stop, off, risk
        else:
            cur_stop = max(entry, peak - trail_mult * atr)
            if L[i] <= cur_stop:
                total = 0.5 * half_r + 0.5 * ((cur_stop - entry) / risk)
                res = 'win' if total > 0 else ('loss' if total < 0 else 'timeout')
                return res, entry + total * risk, off, risk
    last = min(entry_idx + hold, n - 1)
    rem_r = (C[last] - entry) / risk
    total = rem_r if half_r is None else (0.5 * half_r + 0.5 * rem_r)
    return 'timeout', entry + total * risk, last - entry_idx, risk


def build_exit_variants(hold_bars):
    """Registry of exit strategies to compare on identical entries.

    Each entry is (name, resolve_fn, hold_override). hold_override=None uses the
    base hold horizon; time-stop/long-hold variants override it.
    """
    H = hold_bars
    variants = [('baseline 1.5/3.0', lambda *a: resolve_fixed(*a, stop_mult=1.5, target_mult=3.0), None)]
    for sm in (1.0, 1.5):
        for tm in (2.0, 2.5, 3.0):
            variants.append((f'fixed {sm}/{tm}',
                             (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(sm, tm),
                             None))
    for tr in (2.0, 2.5, 3.0):
        variants.append((f'trail {tr}xATR',
                         (lambda tval: (lambda *a: resolve_trail(*a, trail_mult=tval)))(tr),
                         None))
    variants.append(('scaleout 1.5+trail2.0',
                     lambda *a: resolve_scaleout(*a, trail_mult=2.0, partial_mult=1.5), None))
    variants.append(('timestop 1.5/3.0 (half hold)',
                     lambda *a: resolve_fixed(*a, stop_mult=1.5, target_mult=3.0), max(2, H // 2)))
    variants.append(('longhold 1.5/3.0 (2x hold)',
                     lambda *a: resolve_fixed(*a, stop_mult=1.5, target_mult=3.0), H * 2))
    return variants


# =========================================================================
# Single-window simulation
# =========================================================================

def _window_mask(index, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if index.tz is not None:
        s, e = s.tz_localize(index.tz), e.tz_localize(index.tz)
    return (index >= s) & (index < e)


# Default weights mirror the live compute_signal_score (max 17).
DEFAULT_WEIGHTS = {
    'sma200': 3, 'adx_strong': 2, 'adx_trending': 1, 'rs': 2,
    'rsi_sweet': 3, 'rsi_warm': 1, 'bb_bounce': 3, 'bb_cross': 1,
    'vol_surge': 2, 'vol_above': 1, 'macd_strong': 2, 'macd_improving': 1,
}


def make_param_scorer(weights):
    """Return a scorer with the same interface/details as compute_signal_score but
    using a custom weight map, so weight configurations can be A/B validated."""
    w = weights

    def _score(intra_df, daily_ctx, lookback_bars=5):
        if intra_df is None or intra_df.empty or daily_ctx is None:
            return 0, None
        n = max(3, int(lookback_bars))
        recent = intra_df.tail(n)
        if recent.empty:
            return 0, None
        now = intra_df.iloc[-1]
        score = 0
        details = {}

        if daily_ctx.get('above_sma200', False):
            score += w['sma200']; details['above_sma200'] = True
        else:
            details['above_sma200'] = False

        details['adx'] = daily_ctx.get('adx', 0.0)
        if daily_ctx.get('adx_strong', False):
            score += w['adx_strong']; details['adx_tier'] = 'strong'
        elif daily_ctx.get('adx_trending', False):
            score += w['adx_trending']; details['adx_tier'] = 'trending'
        else:
            details['adx_tier'] = 'weak'

        details['rs_outperforming'] = bool(daily_ctx.get('rs_outperforming', False))
        if details['rs_outperforming']:
            score += w['rs']

        rsi_val = float(now['RSI']) if pd.notna(now.get('RSI')) else np.nan
        rsi_rising = bool(now.get('RSI_Rising')) if pd.notna(now.get('RSI_Rising')) else False
        details['rsi'] = rsi_val; details['rsi_rising'] = rsi_rising
        if pd.notna(rsi_val):
            if 40 <= rsi_val <= 60 and rsi_rising:
                score += w['rsi_sweet']; details['rsi_zone'] = 'sweet_spot'
            elif 60 < rsi_val <= 70:
                score += w['rsi_warm']; details['rsi_zone'] = 'warm'
            else:
                details['rsi_zone'] = 'neutral'
        else:
            details['rsi_zone'] = 'na'

        if 'BB_Lower_Bounce' in recent.columns and recent['BB_Lower_Bounce'].any():
            score += w['bb_bounce']; details['bb_lower_bounce'] = True
        elif 'BB_Buy' in recent.columns and recent['BB_Buy'].any():
            score += w['bb_cross']; details['bb_middle_cross'] = True

        if pd.notna(now.get('Volume_Surge')) and bool(now['Volume_Surge']):
            score += w['vol_surge']; details['volume_surge'] = True
        elif pd.notna(now.get('Volume_Above_Avg')) and bool(now['Volume_Above_Avg']):
            score += w['vol_above']; details['volume_above_avg'] = True

        hist_pos = bool(now.get('MACD_Hist_Positive')) if pd.notna(now.get('MACD_Hist_Positive')) else False
        hist_rising = bool(now.get('MACD_Hist_Rising')) if pd.notna(now.get('MACD_Hist_Rising')) else False
        if hist_pos and hist_rising:
            score += w['macd_strong']; details['macd_hist'] = 'strong'
        elif hist_rising:
            score += w['macd_improving']; details['macd_hist'] = 'improving'
        else:
            details['macd_hist'] = 'neutral'

        price = float(now['Close']) if pd.notna(now.get('Close')) else np.nan
        daily_atr = daily_ctx.get('daily_atr', np.nan)
        if pd.notna(price) and pd.notna(daily_atr) and daily_atr > 0:
            atr_pct = (daily_atr / price) * 100.0
        else:
            atr_pct = np.nan
        bb_pos = float(now['BB_Position']) if pd.notna(now.get('BB_Position')) else np.nan
        details.update({'price': price, 'daily_atr': daily_atr, 'atr_pct': atr_pct,
                        'bb_position': bb_pos})
        return score, details

    return _score


def collect_entries(window, subset, frames, provider, base_threshold, max_trades_per_ticker,
                    earnings_dates=None, hold_bars=0,
                    non_overlapping=False, arrays=None, resolve_fn=None, score_fn=None,
                    sector_map=None, sector_provider=None):
    """Find every fresh threshold-cross in a window and record the entry (next-bar
    open). No exit logic here, so the same entries can be replayed under any exit.
    Each entry is tagged with whether the hold window would span an earnings date.

    If non_overlapping is set, an entry blocks new entries on the same ticker until
    its trade (resolved with resolve_fn) would have exited — the realistic
    one-position-per-ticker rule, vs the overlapping mode used for fair A/B tests."""
    entries = []
    for tkr in subset:
        if tkr not in frames:
            continue
        intra_df = frames[tkr]
        win_mask = _window_mask(intra_df.index, window['start'], window['end'])
        positions = np.where(win_mask)[0]
        if len(positions) == 0:
            continue

        prev_score = 0
        trades_this = 0
        cooldown_until = -1

        for pos in positions:
            if trades_this >= max_trades_per_ticker:
                break
            if pos < cooldown_until:
                continue
            trade_date = intra_df.index[pos]
            ctx = provider.get(tkr, trade_date)
            if ctx is None:
                prev_score = 0
                continue

            scorer = score_fn if score_fn is not None else compute_signal_score
            score, details = scorer(intra_df.iloc[:pos + 1], ctx, lookback_bars=5)
            crossed = (prev_score < base_threshold) and (score >= base_threshold)
            prev_score = score
            if not crossed or details is None:
                continue

            entry_idx = pos + 1  # next-bar-open fill
            if entry_idx >= len(intra_df):
                continue
            entry_price = float(intra_df['Open'].iloc[entry_idx])
            daily_atr = ctx.get('daily_atr', np.nan)
            if not (np.isfinite(entry_price) and np.isfinite(daily_atr) and daily_atr > 0):
                continue

            regime_label, regime_adjust = provider.regime(trade_date)

            earnings_blackout = False
            if earnings_dates and tkr in earnings_dates and earnings_dates[tkr]:
                n_bars = len(intra_df)
                exit_pos = min(entry_idx + hold_bars, n_bars - 1)
                entry_date = intra_df.index[entry_idx].date()
                exit_date = intra_df.index[exit_pos].date()
                earnings_blackout = any(entry_date < ed <= exit_date for ed in earnings_dates[tkr])

            entries.append({
                'window': window['name'], 'split': window['split'], 'ticker': tkr,
                'entry_idx': entry_idx, 'entry_time': str(intra_df.index[entry_idx]),
                'entry_price': entry_price, 'daily_atr': daily_atr, 'score': score,
                'earnings_blackout': earnings_blackout,
                'regime': regime_label, 'effective_threshold': base_threshold + regime_adjust,
                'regime_ok': (regime_label != 'risk_off') and (score >= base_threshold + regime_adjust),
                'rsi': details.get('rsi', np.nan),
                'bb_position': details.get('bb_position', np.nan),
                'atr_pct': details.get('atr_pct', np.nan),
                'stk_above_sma20': ctx.get('above_sma20', False),
                'stk_ret_5d': ctx.get('ret_5d', 0.0),
                'sector': sector_map.get(tkr, 'Unknown') if sector_map else 'Unknown',
                'sector_ret_5d': (sector_provider.get(sector_map.get(tkr, ''), trade_date)
                                  if (sector_map and sector_provider) else np.nan),
                'above_sma200': details.get('above_sma200', False),
                'rs_outperforming': details.get('rs_outperforming', False),
                'adx_tier': details.get('adx_tier', 'weak'),
                'rsi_zone': details.get('rsi_zone', 'na'),
                'volume_surge': details.get('volume_surge', False),
                'bb_lower_bounce': details.get('bb_lower_bounce', False),
                'macd_hist': details.get('macd_hist', 'neutral'),
            })
            trades_this += 1
            if non_overlapping and arrays is not None and resolve_fn is not None and tkr in arrays:
                a = arrays[tkr]
                _, _, bars_held, _ = resolve_fn(
                    a['high'], a['low'], a['close'], entry_idx,
                    entry_price, daily_atr, hold_bars,
                )
                cooldown_until = entry_idx + bars_held + 1
            else:
                cooldown_until = entry_idx + 1  # overlapping: only avoid same-bar restack

    return entries


def resolve_entries(entries, arrays, resolve_fn, hold_bars, slippage_bps):
    """Apply one exit strategy to all entries, with slippage costs, return trades df."""
    slip = slippage_bps / 1e4
    rows = []
    for e in entries:
        a = arrays.get(e['ticker'])
        if a is None:
            continue
        result, exit_px, bars_held, risk = resolve_fn(
            a['high'], a['low'], a['close'], e['entry_idx'],
            e['entry_price'], e['daily_atr'], hold_bars,
        )
        if risk <= 0:
            continue
        entry = e['entry_price']
        gross = (exit_px - entry) / risk
        cost_r = slip * (entry + exit_px) / risk  # slippage both sides, in R units
        r_net = gross - cost_r
        row = dict(e)
        row.update({
            'result': result, 'exit_price': exit_px, 'bars_held': bars_held,
            'r_multiple': r_net, 'pct_return': ((exit_px - entry) / entry) * 100 - slip * 200,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# Metrics + reporting
# =========================================================================

def expectancy_stats(df_t):
    if df_t is None or df_t.empty:
        return None
    n = len(df_t)
    wins = df_t[df_t['result'] == 'win']
    losses = df_t[df_t['result'] == 'loss']
    timeouts = df_t[df_t['result'] == 'timeout']
    r = df_t['r_multiple']
    pos_r = r[r > 0].sum()
    neg_r = r[r < 0].sum()
    profit_factor = (pos_r / abs(neg_r)) if neg_r < 0 else float('inf')
    return {
        'trades': n,
        'win_rate': len(wins) / n * 100,
        'loss_rate': len(losses) / n * 100,
        'timeout_rate': len(timeouts) / n * 100,
        'expectancy': r.mean(),
        'avg_win_r': wins['r_multiple'].mean() if len(wins) else 0.0,
        'avg_loss_r': losses['r_multiple'].mean() if len(losses) else 0.0,
        'profit_factor': profit_factor,
        'total_r': r.sum(),
        'avg_pct': df_t['pct_return'].mean(),
    }


def _print_stats(label, s):
    if s is None:
        print(f"\n  {label}: 0 trades")
        return
    pf = s['profit_factor']
    pf_str = "inf" if pf == float('inf') else f"{pf:.2f}"
    print(f"\n  {label}")
    print(f"  {'-' * 64}")
    print(f"  Trades: {s['trades']:4d}   Win {s['win_rate']:5.1f}%   "
          f"Loss {s['loss_rate']:5.1f}%   Timeout {s['timeout_rate']:5.1f}%")
    print(f"  EXPECTANCY: {s['expectancy']:+.3f} R/trade   "
          f"Profit factor: {pf_str}   Total: {s['total_r']:+.1f} R")
    print(f"  Avg winner: {s['avg_win_r']:+.2f} R   Avg loser: {s['avg_loss_r']:+.2f} R   "
          f"Avg return: {s['avg_pct']:+.2f}%")


def report(df, base_threshold, regime_filtered):
    if df is None or df.empty:
        print("\n  No trades generated. Try a lower --threshold.\n")
        return
    view = df[df['regime_ok']] if regime_filtered else df
    filt_label = "REGIME-FILTERED (live-equivalent)" if regime_filtered else f"ALL (score>={base_threshold})"

    print(f"\n{'=' * 72}")
    print(f"  BASELINE SCORECARD  |  {filt_label}")
    print(f"{'=' * 72}")

    print("\n  --- Per window ---")
    per_window = []
    for name in df['window'].unique():
        wv = view[view['window'] == name]
        s = expectancy_stats(wv)
        split = df[df['window'] == name]['split'].iloc[0]
        if s is None:
            print(f"  {name:14s} [{split:5s}]   0 trades")
            continue
        per_window.append((name, split, s['expectancy']))
        print(f"  {name:14s} [{split:5s}]   n={s['trades']:4d}   "
              f"exp={s['expectancy']:+.3f}R   win={s['win_rate']:5.1f}%   PF={s['profit_factor']:.2f}")

    for split in ['train', 'test']:
        sv = view[view['split'] == split]
        _print_stats(f"AGGREGATE — {split.upper()}", expectancy_stats(sv))

    _print_stats("AGGREGATE — ALL WINDOWS", expectancy_stats(view))

    if per_window:
        exps = [e for _, _, e in per_window]
        print(f"\n  --- Cross-window stability (overfit / regime-fragility check) ---")
        print(f"  Mean window expectancy: {np.mean(exps):+.3f} R")
        print(f"  Worst window:           {np.min(exps):+.3f} R")
        print(f"  Std across windows:     {np.std(exps):.3f} R")
        print(f"  Windows profitable:     {sum(1 for e in exps if e > 0)}/{len(exps)}")

    print(f"\n  --- Performance by score ---")
    for sc in sorted(view['score'].unique()):
        sub = view[view['score'] == sc]
        s = expectancy_stats(sub)
        print(f"    score {sc:2d}: n={s['trades']:4d}  exp={s['expectancy']:+.3f}R  "
              f"win={s['win_rate']:5.1f}%  PF={s['profit_factor']:.2f}")

    out_csv = "backtest_harness_trades.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Full trade log ({len(df)} rows) saved to {out_csv}")
    print(f"{'=' * 72}\n")


def sweep_thresholds(windows, subset, frames, arrays, provider, lo, hi,
                     hold_bars, slippage_bps, max_trades, stop_mult, target_mult,
                     regime_filtered):
    """Re-collect entries at each base threshold and report expectancy vs selectivity."""
    label = "regime-filtered" if regime_filtered else "all signals"
    print(f"\n{'=' * 84}")
    print(f"  THRESHOLD SWEEP  |  exit {stop_mult}/{target_mult}xATR  |  {label}  |  slippage {slippage_bps}bps/side")
    print(f"{'=' * 84}")
    print(f"\n  {'thr':>4} {'n':>6} {'TRAIN exp':>10} {'TEST exp':>10} {'all exp':>9} "
          f"{'PF':>6} {'win%':>6} {'worstW':>8} {'totalR':>8}")
    print(f"  {'-' * 80}")
    for t in range(lo, hi + 1):
        entries = []
        for w in windows:
            entries.extend(collect_entries(w, subset, frames, provider, t, max_trades))
        if not entries:
            print(f"  {t:>4d}      0")
            continue
        df = resolve_entries(
            entries, arrays,
            (lambda s, tg: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=tg)))(stop_mult, target_mult),
            hold_bars, slippage_bps,
        )
        view = df[df['regime_ok']] if regime_filtered else df
        if view.empty:
            print(f"  {t:>4d}      0")
            continue
        train = expectancy_stats(view[view['split'] == 'train'])
        test = expectancy_stats(view[view['split'] == 'test'])
        alls = expectancy_stats(view)
        per_win = [expectancy_stats(view[view['window'] == w])['expectancy']
                   for w in view['window'].unique()
                   if expectancy_stats(view[view['window'] == w])]
        worst = min(per_win) if per_win else float('nan')
        pf = "inf" if alls['profit_factor'] == float('inf') else f"{alls['profit_factor']:.2f}"
        tr = f"{train['expectancy']:+.3f}" if train else "   n/a"
        te = f"{test['expectancy']:+.3f}" if test else "   n/a"
        print(f"  {t:>4d} {alls['trades']:>6d} {tr:>10} {te:>10} {alls['expectancy']:>+9.3f} "
              f"{pf:>6} {alls['win_rate']:>5.1f}% {worst:>+8.3f} {alls['total_r']:>+8.1f}")
    print(f"\n  Higher threshold = fewer, higher-quality trades. Pick the knee where")
    print(f"  expectancy is strong on BOTH train+test and total-R hasn't collapsed.")
    print(f"{'=' * 84}\n")


# Data-driven candidate: drop the inverted BB-bounce bonus and near-useless ADX
# weight, zero the underperforming RSI-warm band, and boost volume (a real
# discriminator). Derived from the train-split feature impact.
CANDIDATE_WEIGHTS = {
    'sma200': 3, 'adx_strong': 1, 'adx_trending': 1, 'rs': 1,
    'rsi_sweet': 3, 'rsi_warm': 0, 'bb_bounce': 1, 'bb_cross': 1,
    'vol_surge': 3, 'vol_above': 1, 'macd_strong': 1, 'macd_improving': 1,
}


def entry_timing_test(entries, arrays, hold_bars, slippage_bps, stop_mult, target_mult,
                      regime_filtered, wait_bars):
    """Chase-at-alert vs retrace-limit entries on identical signals.
    Limit fills pay no entry slippage (passive side); unfilled signals are skipped."""
    slip = slippage_bps / 1e4
    variants = [('CHASE next-bar open (baseline)', None)]
    variants += [(f'LIMIT {k:.2f}xATR below, wait {wait_bars}b', k) for k in (0.15, 0.25, 0.40)]

    print(f"\n{'=' * 96}")
    print(f"  ENTRY TIMING TEST  |  exit {stop_mult}/{target_mult}xATR  |  "
          f"{'regime-filtered' if regime_filtered else 'all'}  |  {len(entries)} signals")
    print(f"  {'variant':34s} {'fill%':>6} {'n':>5} {'exp/fill':>9} {'exp/signal':>10} "
          f"{'TEST exp':>9} {'win%':>6} {'PF':>6}")
    print(f"{'=' * 96}")

    for name, k in variants:
        rs, test_rs, wins, gains, losses = [], [], 0, 0.0, 0.0
        n_signals = n_filled = 0
        for e in entries:
            if regime_filtered and not e['regime_ok']:
                continue
            a = arrays.get(e['ticker'])
            if a is None or e['daily_atr'] <= 0:
                continue
            n_signals += 1
            H, L, C, O = a['high'], a['low'], a['close'], a['open']
            atr = e['daily_atr']
            if k is None:
                fill_idx, fill_px, entry_slip = e['entry_idx'], e['entry_price'], slip
            else:
                limit = e['entry_price'] - k * atr
                fill_idx = fill_px = None
                for off in range(0, wait_bars + 1):
                    i = e['entry_idx'] + off
                    if i >= len(C):
                        break
                    if O[i] <= limit:
                        fill_idx, fill_px = i, O[i]
                        break
                    if L[i] <= limit:
                        fill_idx, fill_px = i, limit
                        break
                if fill_idx is None:
                    continue
                entry_slip = 0.0
            n_filled += 1
            result, exit_px, _, risk = resolve_fixed(
                H, L, C, fill_idx, fill_px, atr, hold_bars, stop_mult, target_mult)
            if risk <= 0:
                continue
            r = (exit_px - fill_px) / risk - (entry_slip * fill_px + slip * exit_px) / risk
            rs.append(r)
            if e['split'] == 'test':
                test_rs.append(r)
            if r > 0:
                wins += 1
                gains += r
            else:
                losses += -r
        if not rs:
            print(f"  {name:34s}   no fills")
            continue
        te = f"{np.mean(test_rs):+.3f}" if test_rs else "  n/a"
        pf = gains / losses if losses > 0 else float('inf')
        exp_fill = np.mean(rs)
        exp_signal = np.sum(rs) / max(n_signals, 1)  # unfilled = 0R opportunity
        print(f"  {name:34s} {n_filled / max(n_signals, 1) * 100:5.1f}% {len(rs):5d} "
              f"{exp_fill:>+9.3f} {exp_signal:>+10.3f} {te:>9} "
              f"{wins / len(rs) * 100:5.1f}% {pf:6.2f}")
    print(f"{'=' * 96}")
    print("  exp/fill = expectancy per filled trade; exp/signal = per alert incl. unfilled (0R).\n")


def sector_regime_analysis(entries, arrays, hold_bars, slippage_bps, stop_mult, target_mult,
                           regime_filtered):
    """Validate: (a) sector-ETF momentum veto, (b) per-day sector caps,
    (c) regime-dependent exits (tighter target in caution)."""
    mk = lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t))
    df3 = resolve_entries(entries, arrays, mk(stop_mult, target_mult), hold_bars, slippage_bps)
    df2 = resolve_entries(entries, arrays, mk(stop_mult, 2.0), hold_bars, slippage_bps)
    if df3.empty:
        print("\n  No trades.\n")
        return
    if regime_filtered:
        keep = df3['regime_ok']
        df3, df2 = df3[keep], df2[keep.values]
    for d in (df3, df2):
        d['date'] = pd.to_datetime(d['entry_time']).dt.date

    print(f"\n{'=' * 78}")
    print(f"  SECTOR & REGIME ANALYSIS  |  exit {stop_mult}/{target_mult}xATR  |  "
          f"{'regime-filtered' if regime_filtered else 'all'}")
    print(f"{'=' * 78}")

    def _line(name, sub):
        s = expectancy_stats(sub)
        if not s:
            print(f"    {name:30s}    0 trades")
            return
        ts = expectancy_stats(sub[sub['split'] == 'test'])
        te = f"{ts['expectancy']:+.3f}" if ts else "  n/a"
        print(f"    {name:30s} n={s['trades']:5d}  exp={s['expectancy']:+.3f}R "
              f"(test {te})  win={s['win_rate']:4.1f}%  PF={s['profit_factor']:.2f}")

    print(f"\n  --- (a) by sector-ETF 5d return at entry ---")
    col = df3['sector_ret_5d']
    for lo, hi in [(-100, -5), (-5, -3), (-3, 0), (0, 3), (3, 100)]:
        _line(f"sector 5d in [{lo:g}, {hi:g})", df3[(col >= lo) & (col < hi)])
    for thr in (-3.0, -4.0, -5.0):
        _line(f"VETO if sector 5d <= {thr:g}%", df3[~(col <= thr)])

    print(f"\n  --- (b) per-day sector cap (chronological first-N) ---")
    _line("no cap", df3)
    for cap in (3, 2, 1):
        capped = (df3.sort_values('entry_time')
                  .groupby(['date', 'sector']).head(cap))
        _line(f"max {cap}/sector/day", capped)

    print(f"\n  --- (c) regime-dependent exits ---")
    caution3 = df3[df3['regime'] == 'caution']
    caution2 = df2[df2['regime'] == 'caution']
    riskon3 = df3[df3['regime'] == 'risk_on']
    _line("risk_on trades @ 3x target", riskon3)
    _line("caution trades @ 3x target", caution3)
    _line("caution trades @ 2x target", caution2)
    _line("caution trades SKIPPED", riskon3)
    blended = pd.concat([riskon3, caution2])
    _line("BLEND: 3x risk_on + 2x caution", blended)
    _line("baseline: 3x everywhere", df3)
    print(f"{'=' * 78}\n")


def topn_test(entries, arrays, hold_bars, slippage_bps, stop_mult, target_mult, regime_filtered):
    """Does taking only the top-N highest-scoring signals per day concentrate the edge?"""
    df = resolve_entries(
        entries, arrays,
        (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(stop_mult, target_mult),
        hold_bars, slippage_bps,
    )
    if df.empty:
        print("\n  No trades.\n")
        return
    if regime_filtered:
        df = df[df['regime_ok']]
    df = df.copy()
    df['date'] = pd.to_datetime(df['entry_time']).dt.date

    print(f"\n{'=' * 72}")
    print(f"  TOP-N PER DAY RANKING  |  exit {stop_mult}/{target_mult}xATR  |  "
          f"{'regime-filtered' if regime_filtered else 'all'}")
    print(f"  (rank each day's signals by score, take the best N; ties broken by volume_surge)")
    print(f"{'=' * 72}")
    df['rank_key'] = df['score'] + df['volume_surge'].astype(float) * 0.5
    print(f"\n  {'N/day':>7} {'trades':>7} {'exp':>9} {'test exp':>9} {'PF':>6} {'win%':>6} {'avg/day':>8}")
    print(f"  {'-' * 60}")
    n_days = df['date'].nunique()
    for n in [1, 2, 3, 5, 10, 9999]:
        picks = df.sort_values('rank_key', ascending=False).groupby('date').head(n)
        s = expectancy_stats(picks)
        ts = expectancy_stats(picks[picks['split'] == 'test'])
        te = f"{ts['expectancy']:+.3f}" if ts else "  n/a"
        nd = "all" if n == 9999 else str(n)
        print(f"  {nd:>7} {s['trades']:>7d} {s['expectancy']:>+9.3f} {te:>9} "
              f"{s['profit_factor']:>6.2f} {s['win_rate']:>5.1f}% {s['trades']/max(n_days,1):>8.1f}")
    print(f"\n  {n_days} trading days in sample. Higher exp at small N = ranking concentrates edge.")
    print(f"{'=' * 72}\n")


def reweight_test(windows, subset, frames, arrays, provider, hold_bars, slippage_bps,
                  stop_mult, target_mult, regime_filtered, max_trades, non_overlapping):
    """Sweep thresholds for default vs candidate weight maps; fit on train, judge on test."""
    adopted_exit = (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(
        stop_mult, target_mult)
    configs = [('DEFAULT (max17)', DEFAULT_WEIGHTS, range(8, 14)),
              ('CANDIDATE (max13)', CANDIDATE_WEIGHTS, range(6, 12))]

    for name, weights, thr_range in configs:
        scorer = make_param_scorer(weights)
        print(f"\n{'=' * 80}")
        print(f"  REWEIGHT TEST — {name}  |  exit {stop_mult}/{target_mult}xATR  |  "
              f"{'regime-filtered' if regime_filtered else 'all'}")
        print(f"  {'thr':>4} {'n':>6} {'TRAIN exp':>10} {'TEST exp':>10} {'all exp':>9} {'PF':>6} {'win%':>6}")
        print(f"  {'-' * 70}")
        for t in thr_range:
            entries = []
            for w in windows:
                entries.extend(collect_entries(
                    w, subset, frames, provider, t, max_trades,
                    hold_bars=hold_bars, non_overlapping=non_overlapping,
                    arrays=arrays, resolve_fn=adopted_exit, score_fn=scorer))
            if not entries:
                print(f"  {t:>4d}      0")
                continue
            df = resolve_entries(entries, arrays, adopted_exit, hold_bars, slippage_bps)
            view = df[df['regime_ok']] if regime_filtered else df
            if view.empty:
                print(f"  {t:>4d}      0")
                continue
            tr = expectancy_stats(view[view['split'] == 'train'])
            te = expectancy_stats(view[view['split'] == 'test'])
            al = expectancy_stats(view)
            trs = f"{tr['expectancy']:+.3f}" if tr else "   n/a"
            tes = f"{te['expectancy']:+.3f}" if te else "   n/a"
            print(f"  {t:>4d} {al['trades']:>6d} {trs:>10} {tes:>10} {al['expectancy']:>+9.3f} "
                  f"{al['profit_factor']:>6.2f} {al['win_rate']:>5.1f}%")
    print(f"\n{'=' * 80}\n")


def analyze_filters(entries, arrays, hold_bars, slippage_bps, stop_mult, target_mult,
                    regime_filtered):
    """Resolve entries once, then show expectancy bucketed by numeric features and
    categorical flags — the data that drives Step 4 filters and category re-weighting."""
    df = resolve_entries(
        entries, arrays,
        (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(stop_mult, target_mult),
        hold_bars, slippage_bps,
    )
    if df.empty:
        print("\n  No trades.\n")
        return
    if regime_filtered:
        df = df[df['regime_ok']]
    label = "regime-filtered" if regime_filtered else "all signals"
    print(f"\n{'=' * 78}")
    print(f"  FILTER / FEATURE IMPACT  |  exit {stop_mult}/{target_mult}xATR  |  {label}")
    print(f"  (test-split expectancy in parens to check robustness)")
    print(f"{'=' * 78}")

    def _line(name, sub):
        s = expectancy_stats(sub)
        if not s:
            print(f"    {name:24s}    0 trades")
            return
        ts = expectancy_stats(sub[sub['split'] == 'test'])
        te = f"{ts['expectancy']:+.3f}" if ts else "  n/a"
        print(f"    {name:24s} n={s['trades']:5d}  exp={s['expectancy']:+.3f}R "
              f"(test {te})  win={s['win_rate']:4.1f}%  PF={s['profit_factor']:.2f}")

    def _numeric(field, edges):
        print(f"\n  --- by {field} ---")
        col = df[field]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = df[(col >= lo) & (col < hi)]
            _line(f"[{lo:g}, {hi:g})", sub)

    _numeric('bb_position', [-50, 0, 20, 40, 60, 80, 100, 200])
    _numeric('atr_pct', [0, 1, 2, 3, 4, 6, 100])
    _numeric('rsi', [0, 30, 40, 50, 60, 70, 100])

    for cat in ['adx_tier', 'rsi_zone', 'macd_hist']:
        print(f"\n  --- by {cat} ---")
        for val in df[cat].dropna().unique():
            _line(str(val), df[df[cat] == val])

    for flag in ['above_sma200', 'rs_outperforming', 'volume_surge', 'bb_lower_bounce']:
        if flag not in df.columns:
            continue
        print(f"\n  --- {flag} ---")
        _line("True", df[df[flag] == True])
        _line("False", df[df[flag] != True])

    print(f"\n  --- candidate Step-4 entry filters (vs unfiltered baseline) ---")
    _line("baseline (no filter)", df)
    _line("atr_pct >= 1.0", df[df['atr_pct'] >= 1.0])
    _line("bb_position <= 90", df[df['bb_position'] <= 90])
    _line("rsi <= 70", df[df['rsi'] <= 70])
    combo = df[(df['atr_pct'] >= 1.0) & (df['bb_position'] <= 90) & (df['rsi'] <= 70)]
    _line("ALL three combined", combo)

    print(f"\n  --- falling-knife guard (stock-level short-term momentum) ---")
    _numeric('stk_ret_5d', [-100, -10, -6, -3, 0, 3, 6, 100])
    print()
    _line("stk 5d ret > -6%", df[df['stk_ret_5d'] > -6.0])
    _line("stk 5d ret > -3%", df[df['stk_ret_5d'] > -3.0])
    _line("stk above SMA20", df[df['stk_above_sma20'] == True])
    _line("stk BELOW SMA20", df[df['stk_above_sma20'] != True])
    knife = df[(df['stk_ret_5d'] > -6.0) & (df['stk_above_sma20'] == True)]
    _line("ret>-6% AND >SMA20", knife)
    _line("band -6% < ret < +6%", df[(df['stk_ret_5d'] > -6.0) & (df['stk_ret_5d'] < 6.0)])

    for lo_sc in (11, 14):
        hi = df[df['score'] >= lo_sc]
        print(f"\n  score >= {lo_sc} slice:")
        _line("  baseline", hi)
        _line("  + knife (ret>-6%)", hi[hi['stk_ret_5d'] > -6.0])
        _line("  + band (-6..+6%)", hi[(hi['stk_ret_5d'] > -6.0) & (hi['stk_ret_5d'] < 6.0)])

    print(f"{'=' * 78}\n")


def compare_earnings_blackout(entries, arrays, hold_bars, slippage_bps,
                              stop_mult, target_mult, regime_filtered):
    """Resolve all entries once, then split by whether the hold spans earnings."""
    df = resolve_entries(
        entries, arrays,
        (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(stop_mult, target_mult),
        hold_bars, slippage_bps,
    )
    if df.empty:
        print("\n  No trades.\n")
        return
    if regime_filtered:
        df = df[df['regime_ok']]
    label = "regime-filtered" if regime_filtered else "all signals"
    print(f"\n{'=' * 78}")
    print(f"  EARNINGS BLACKOUT IMPACT  |  exit {stop_mult}/{target_mult}xATR  |  {label}")
    print(f"{'=' * 78}")

    held_through = df[df['earnings_blackout']]
    clean = df[~df['earnings_blackout']]
    _print_stats("HELD THROUGH EARNINGS (would be vetoed)", expectancy_stats(held_through))
    _print_stats("CLEAN — no earnings in hold window (kept)", expectancy_stats(clean))
    _print_stats("ALL (current behavior, no blackout)", expectancy_stats(df))

    print(f"\n  --- Out-of-sample check (kept vs all) ---")
    for split in ['train', 'test']:
        a = expectancy_stats(df[df['split'] == split])
        k = expectancy_stats(clean[clean['split'] == split])
        if a and k:
            print(f"  {split:5s}:  ALL exp {a['expectancy']:+.3f}R (n={a['trades']})"
                  f"   ->  BLACKOUT-FILTERED exp {k['expectancy']:+.3f}R (n={k['trades']})")
    print(f"{'=' * 78}\n")


def compare_exits(entries, arrays, hold_bars, slippage_bps, regime_filtered):
    """Resolve all entries under every exit variant and rank by expectancy."""
    variants = build_exit_variants(hold_bars)
    label = "regime-filtered (live-equivalent)" if regime_filtered else "all signals"
    print(f"\n{'=' * 84}")
    print(f"  EXIT-STRATEGY COMPARISON  |  {label}  |  same entries, slippage {slippage_bps}bps/side")
    print(f"{'=' * 84}")
    print(f"\n  {'variant':<28} {'n':>5} {'TRAIN exp':>10} {'TEST exp':>10} "
          f"{'all exp':>9} {'PF':>6} {'win%':>6} {'worstW':>8}")
    print(f"  {'-' * 82}")

    results = []
    for name, fn, hold_override in variants:
        hold = hold_override if hold_override is not None else hold_bars
        df = resolve_entries(entries, arrays, fn, hold, slippage_bps)
        if df.empty:
            continue
        view = df[df['regime_ok']] if regime_filtered else df
        if view.empty:
            continue
        train = expectancy_stats(view[view['split'] == 'train'])
        test = expectancy_stats(view[view['split'] == 'test'])
        alls = expectancy_stats(view)
        per_win = [expectancy_stats(view[view['window'] == w])['expectancy']
                   for w in view['window'].unique()
                   if expectancy_stats(view[view['window'] == w])]
        worst = min(per_win) if per_win else float('nan')
        results.append({
            'name': name, 'n': alls['trades'],
            'train_exp': train['expectancy'] if train else float('nan'),
            'test_exp': test['expectancy'] if test else float('nan'),
            'all_exp': alls['expectancy'], 'pf': alls['profit_factor'],
            'win': alls['win_rate'], 'worst': worst,
        })

    results.sort(key=lambda r: (r['train_exp'] if np.isfinite(r['train_exp']) else -9), reverse=True)
    for r in results:
        pf = "inf" if r['pf'] == float('inf') else f"{r['pf']:.2f}"
        print(f"  {r['name']:<28} {r['n']:>5d} {r['train_exp']:>+10.3f} {r['test_exp']:>+10.3f} "
              f"{r['all_exp']:>+9.3f} {pf:>6} {r['win']:>5.1f}% {r['worst']:>+8.3f}")

    print(f"\n  Ranked by TRAIN expectancy. A robust winner has high TRAIN *and* TEST exp,")
    print(f"  a healthy worst-window, and isn't a single-window fluke.")
    print(f"{'=' * 84}\n")
    return results


# =========================================================================
# Main
# =========================================================================

# =========================================================================
# Portfolio benchmark: does the signal beat simply owning the index?
# =========================================================================
# expectancy_stats measures R per trade, which can be positive while the
# strategy still trails buy-and-hold: R ignores how much capital was tied up,
# how long, and how many signals a small account could actually take. These
# functions answer the only question that decides whether to trade the system:
# after capping concurrent positions, did it beat SPY, and did it beat picking
# the same number of names at random on the same days?

def build_timeline(entries, arrays, resolve_fn, hold_bars, slippage_bps):
    """Resolve entries into trades carrying wall-clock entry/exit timestamps.

    resolve_entries returns positional bar offsets, which are enough for R stats
    but not for a portfolio: overlapping positions have to compete for slots in
    real time, so exits need timestamps that are comparable across tickers.
    """
    slip = slippage_bps / 1e4
    rows = []
    for e in entries:
        a = arrays.get(e['ticker'])
        if a is None:
            continue
        result, exit_px, bars_held, risk = resolve_fn(
            a['high'], a['low'], a['close'], e['entry_idx'],
            e['entry_price'], e['daily_atr'], hold_bars,
        )
        if risk <= 0:
            continue
        idx = a['index']
        exit_pos = min(e['entry_idx'] + bars_held, len(idx) - 1)
        entry = e['entry_price']
        row = dict(e)
        row.update({
            'result': result, 'exit_price': exit_px, 'bars_held': bars_held,
            'entry_ts': idx[e['entry_idx']], 'exit_ts': idx[exit_pos],
            'r_multiple': (exit_px - entry) / risk - slip * (entry + exit_px) / risk,
            'pct_return': ((exit_px - entry) / entry) * 100 - slip * 200,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_portfolio(trades, max_positions):
    """Equal-weight portfolio with a hard cap on concurrent positions.

    Signals arriving while every slot is full are skipped, not queued - that is
    what actually happens to an account holding 1-5 names. Returns realized
    equity, so a strategy that ties up capital in slow trades is penalized the
    way it is penalized in reality.
    """
    if trades is None or trades.empty:
        return None
    t = trades.sort_values('entry_ts')
    cash = 1.0
    open_pos = []           # (exit_ts, size, pct_return)
    curve = [(t['entry_ts'].iloc[0], 1.0)]
    taken = skipped = 0
    slot_time = pd.Timedelta(0)

    def close_due(now):
        nonlocal cash, open_pos
        due = [p for p in open_pos if p[0] <= now]
        if not due:
            return
        open_pos = [p for p in open_pos if p[0] > now]
        held = sum(s for _, s, _ in open_pos)
        pending = sum(s for _, s, _ in due)
        for ex_ts, size, pct in sorted(due, key=lambda x: x[0]):
            cash += size * (1 + pct / 100.0)
            pending -= size
            curve.append((ex_ts, cash + held + pending))

    for _, r in t.iterrows():
        close_due(r['entry_ts'])
        equity_now = cash + sum(s for _, s, _ in open_pos)
        size = equity_now / max_positions
        if len(open_pos) >= max_positions or size > cash + 1e-12:
            skipped += 1
            continue
        cash -= size
        open_pos.append((r['exit_ts'], size, r['pct_return']))
        slot_time += (r['exit_ts'] - r['entry_ts'])
        taken += 1

    remaining = sorted(open_pos, key=lambda x: x[0])
    held = sum(s for _, s, _ in remaining)
    for ex_ts, size, pct in remaining:
        cash += size * (1 + pct / 100.0)
        held -= size
        curve.append((ex_ts, cash + held))

    eq = pd.Series([v for _, v in curve], index=[ts for ts, _ in curve]).sort_index()
    peak = eq.cummax()
    span = t['exit_ts'].max() - t['entry_ts'].min()
    utilization = (slot_time / (span * max_positions) * 100.0) if span.total_seconds() > 0 else np.nan
    return {
        'total_return': (cash - 1.0) * 100.0,
        'max_dd': ((eq / peak - 1.0).min()) * 100.0,
        'taken': taken, 'skipped': skipped, 'signals': len(t),
        'utilization': utilization,
        'equity': eq,
    }


def _spy_window_return(spy_close, start, end):
    s = spy_close.loc[(spy_close.index >= pd.Timestamp(start)) & (spy_close.index <= pd.Timestamp(end))]
    if len(s) < 2:
        return np.nan
    return (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0


def random_control(entries, arrays, provider, subset, resolve_fn, hold_bars,
                   slippage_bps, max_positions, runs=20, seed=0):
    """Same entry times and exit rules, random tickers.

    This is the control that matters. If the real signal cannot beat a coin flip
    taking positions on the same days, the scoring logic is decoration - the
    returns are coming from being long stocks, not from stock selection.
    """
    rng = np.random.default_rng(seed)
    pool = [t for t in subset if t in arrays]
    if not pool:
        return None
    results = []
    for _ in range(runs):
        fake = []
        for e in entries:
            src = arrays.get(e['ticker'])
            if src is None:
                continue
            when = src['index'][e['entry_idx']]
            tkr = pool[rng.integers(len(pool))]
            a = arrays[tkr]
            pos = int(np.searchsorted(a['index'], when))
            if pos >= len(a['index']) - 1:
                continue
            ctx = provider.get(tkr, a['index'][pos].date())
            if ctx is None:
                continue
            atr = ctx.get('daily_atr', np.nan)
            px = float(a['open'][pos])
            if not (np.isfinite(atr) and atr > 0 and np.isfinite(px)):
                continue
            fake.append({'ticker': tkr, 'entry_idx': pos, 'entry_price': px,
                         'daily_atr': atr, 'window': e['window'], 'split': e['split']})
        tl = build_timeline(fake, arrays, resolve_fn, hold_bars, slippage_bps)
        sim = simulate_portfolio(tl, max_positions)
        if sim:
            results.append(sim['total_return'])
    if not results:
        return None
    return {'mean': float(np.mean(results)), 'std': float(np.std(results)),
            'lo': float(np.min(results)), 'hi': float(np.max(results)), 'runs': len(results)}


def benchmark_test(entries, arrays, provider, subset, spy_close, windows, hold_bars,
                   slippage_bps, stop_mult, target_mult, regime_filtered,
                   max_positions, band, runs):
    resolve_fn = (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(
        stop_mult, target_mult)

    ent = list(entries)
    if regime_filtered:
        ent = [e for e in ent if e.get('regime_ok')]
    if band is not None:
        lo, hi = band
        ent = [e for e in ent if lo <= e.get('stk_ret_5d', 0.0) <= hi]

    print(f"\n{'=' * 78}")
    print("  PORTFOLIO BENCHMARK - does this beat owning the index?")
    print(f"  max concurrent positions: {max_positions}   exit: {stop_mult}x/{target_mult}x ATR")
    print(f"  filters: regime={'on' if regime_filtered else 'off'}  "
          f"momentum band={f'{band[0]}..{band[1]}%' if band else 'off'}")
    print(f"{'=' * 78}")

    tl_all = build_timeline(ent, arrays, resolve_fn, hold_bars, slippage_bps)
    if tl_all.empty:
        print("  no trades")
        return

    hdr = (f"\n  {'window':14s} {'split':5s} {'signals':>8s} {'taken':>6s} {'used%':>6s} "
           f"{'strategy':>9s} {'SPY':>8s} {'edge':>8s} {'random':>16s} {'realDD':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    rows = []
    for w in windows:
        sub = tl_all[tl_all['window'] == w['name']]
        if sub.empty:
            continue
        sim = simulate_portfolio(sub, max_positions)
        spy_r = _spy_window_return(spy_close, w['start'], w['end'])
        went = [e for e in ent if e['window'] == w['name']]
        rc = random_control(went, arrays, provider, subset, resolve_fn, hold_bars,
                            slippage_bps, max_positions, runs=runs, seed=abs(hash(w['name'])) % 10000)
        rand_s = f"{rc['mean']:+6.2f}±{rc['std']:4.2f}" if rc else "     n/a"
        print(f"  {w['name']:14s} {w['split']:5s} {sim['signals']:8d} {sim['taken']:6d} "
              f"{sim['utilization']:5.0f}% {sim['total_return']:+8.2f}% {spy_r:+7.2f}% "
              f"{sim['total_return'] - spy_r:+7.2f}pp {rand_s:>16s} {sim['max_dd']:6.1f}%")
        rows.append({'window': w['name'], 'split': w['split'], 'strat': sim['total_return'],
                     'spy': spy_r, 'edge': sim['total_return'] - spy_r,
                     'rand': rc['mean'] if rc else np.nan,
                     'rand_hi': rc['hi'] if rc else np.nan,
                     'rand_lo': rc['lo'] if rc else np.nan})

    if not rows:
        return
    df = pd.DataFrame(rows)
    print()
    for split in ('train', 'test'):
        d = df[df['split'] == split]
        if d.empty:
            continue
        print(f"  {split.upper():5s}  strategy {d['strat'].mean():+6.2f}%   "
              f"SPY {d['spy'].mean():+6.2f}%   edge {d['edge'].mean():+6.2f}pp   "
              f"beat SPY in {int((d['edge'] > 0).sum())}/{len(d)} windows   "
              f"beat random in {int((d['strat'] > d['rand']).sum())}/{len(d)}")

    print(f"\n  ALL    strategy {df['strat'].mean():+6.2f}%   SPY {df['spy'].mean():+6.2f}%   "
          f"edge {df['edge'].mean():+6.2f}pp   beat SPY in {int((df['edge'] > 0).sum())}/{len(df)} windows")
    luck = (df['rand_hi'] - df['rand_lo']).mean()
    print(f"  Random-selection spread across windows: {luck:.2f}pp average "
          f"(best minus worst draw) - an edge smaller than this is not distinguishable from luck.")


def build_exit_grid(hold_bars):
    """Wider exit grid for the portfolio sweep.

    compare_exits ranks these by R-expectancy. This grid exists to rank them by
    portfolio return against SPY instead, which is a different question: a tight
    stop can raise R while parking capital in trades that never resolve.
    Hold length is swept too, since 42% of trades exit on the time stop.
    """
    grid = []
    for sm in (0.75, 1.0, 1.5, 2.0):
        for tm in (1.5, 2.0, 3.0):
            for hname, hmult in (('.5x', 0.5), ('1x', 1.0), ('2x', 2.0)):
                grid.append((
                    f'fixed {sm}/{tm} hold{hname}',
                    (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(sm, tm),
                    max(2, int(hold_bars * hmult)),
                ))
    for tr in (1.5, 2.0, 2.5, 3.0):
        grid.append((f'trail {tr}xATR',
                     (lambda v: (lambda *a: resolve_trail(*a, trail_mult=v)))(tr), hold_bars))
    grid.append(('scaleout 1.5+trail2.0',
                 lambda *a: resolve_scaleout(*a, trail_mult=2.0, partial_mult=1.5), hold_bars))
    return grid


def exit_benchmark(entries, arrays, spy_close, windows, hold_bars, slippage_bps,
                   regime_filtered, band, positions=(3, 5)):
    """Rank exit strategies by portfolio return vs SPY, choosing on train windows only.

    The adopted 1.0x/3.0x exit was selected by maximizing R-expectancy, which the
    portfolio benchmark showed can be positive while the strategy trails the index.
    This re-runs that selection under the metric that actually decides whether the
    system deserves capital.
    """
    ent = list(entries)
    if regime_filtered:
        ent = [e for e in ent if e.get('regime_ok')]
    if band is not None:
        lo, hi = band
        ent = [e for e in ent if lo <= e.get('stk_ret_5d', 0.0) <= hi]

    spy_by_w = {w['name']: _spy_window_return(spy_close, w['start'], w['end']) for w in windows}
    train_w = [w['name'] for w in windows if w['split'] == 'train']
    test_w = [w['name'] for w in windows if w['split'] == 'test']

    grid = build_exit_grid(hold_bars)
    print(f"\n{'=' * 92}")
    print("  EXIT SELECTION BY PORTFOLIO RETURN vs SPY")
    print(f"  {len(grid)} variants x {len(windows)} windows, positions={positions}")
    print("  Selection is on TRAIN windows only; test column is shown for the finalists.")
    print(f"{'=' * 92}")

    rows = []
    for i, (name, fn, hold) in enumerate(grid, 1):
        print(f"\r  evaluating {i}/{len(grid)}: {name:34s}", end='', flush=True)
        tl = build_timeline(ent, arrays, fn, hold, slippage_bps)
        if tl.empty:
            continue
        rec = {'variant': name}
        ok = True
        for p in positions:
            edges = {}
            for w in windows:
                sub = tl[tl['window'] == w['name']]
                sim = simulate_portfolio(sub, p)
                if sim is None or not np.isfinite(spy_by_w[w['name']]):
                    continue
                edges[w['name']] = sim['total_return'] - spy_by_w[w['name']]
            if not edges:
                ok = False
                break
            tr = [edges[n] for n in train_w if n in edges]
            te = [edges[n] for n in test_w if n in edges]
            rec[f'train{p}'] = float(np.mean(tr)) if tr else np.nan
            rec[f'test{p}'] = float(np.mean(te)) if te else np.nan
            rec[f'won{p}'] = int(sum(1 for x in tr if x > 0))
            rec[f'n{p}'] = len(tr)
        if ok:
            rows.append(rec)
    print("\r" + " " * 60 + "\r", end='')

    if not rows:
        print("  no results")
        return
    df = pd.DataFrame(rows)
    p_lo, p_hi = positions[0], positions[-1]
    # Rank on the tighter position count: that is how the account is actually run,
    # and it is the setting where the adopted exit failed.
    df['robust'] = df[[f'train{p}' for p in positions]].min(axis=1)
    df = df.sort_values(f'train{p_lo}', ascending=False)

    print(f"\n  {'variant':34s} {'train@' + str(p_lo):>9s} {'won':>5s} "
          f"{'train@' + str(p_hi):>9s} {'won':>5s} {'worst':>8s}")
    print("  " + "-" * 78)
    for _, r in df.head(14).iterrows():
        print(f"  {r['variant']:34s} {r[f'train{p_lo}']:+8.2f}pp "
              f"{int(r[f'won{p_lo}'])}/{int(r[f'n{p_lo}'])} "
              f"{r[f'train{p_hi}']:+8.2f}pp {int(r[f'won{p_hi}'])}/{int(r[f'n{p_hi}'])} "
              f"{r['robust']:+7.2f}pp")

    adopted = df[df['variant'].str.startswith('fixed 1.0/3.0 hold1x')]
    if not adopted.empty:
        a = adopted.iloc[0]
        print(f"\n  Currently adopted (fixed 1.0/3.0 hold1x): train@{p_lo} {a[f'train{p_lo}']:+.2f}pp, "
              f"train@{p_hi} {a[f'train{p_hi}']:+.2f}pp, "
              f"rank {int(df.index.get_loc(a.name)) + 1} of {len(df)}")

    finalists = df[df['robust'] > 0].head(5)
    print(f"\n  Finalists (positive at every position count), test window revealed:")
    if finalists.empty:
        print("    NONE - no exit variant beats SPY across train windows at both position counts.")
    else:
        for _, r in finalists.iterrows():
            print(f"    {r['variant']:34s} train@{p_lo} {r[f'train{p_lo}']:+6.2f}pp  "
                  f"train@{p_hi} {r[f'train{p_hi}']:+6.2f}pp  ->  "
                  f"TEST@{p_lo} {r[f'test{p_lo}']:+7.2f}pp  TEST@{p_hi} {r[f'test{p_hi}']:+7.2f}pp")
    print("\n  Note: entries were collected under the adopted exit's hold, so non-overlapping\n"
          "  spacing is held fixed across variants - exits are compared on identical entries.")


def main():
    parser = argparse.ArgumentParser(description="Hardened multi-period backtest harness")
    parser.add_argument("--interval", default="1h", help="Bar interval (default 1h; 15m = recent confirmation)")
    parser.add_argument("--threshold", type=int, default=9, help="Base score threshold (default 9)")
    parser.add_argument("--hold-days", type=int, default=5, help="Max hold horizon in trading days (default 5)")
    parser.add_argument("--subset-size", type=int, default=80, help="Liquid subset size (default 80)")
    parser.add_argument("--max-trades", type=int, default=3, help="Max trades per ticker per window")
    parser.add_argument("--refresh", action="store_true", help="Ignore on-disk data cache and re-download")
    parser.add_argument("--regime-filtered", action="store_true",
                        help="Report only trades the live kill-switch would allow")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="Slippage per side in basis points (default 5 = 0.05%%)")
    parser.add_argument("--compare-exits", action="store_true",
                        help="Compare all exit strategies on identical entries")
    parser.add_argument("--stop-mult", type=float, default=1.0,
                        help="ATR stop multiple for the baseline exit (adopted: 1.0)")
    parser.add_argument("--target-mult", type=float, default=3.0,
                        help="ATR target multiple for the baseline exit (adopted: 3.0)")
    parser.add_argument("--sweep-thresholds", type=str, default=None,
                        help="Sweep base thresholds, e.g. '8-14'")
    parser.add_argument("--earnings-blackout", action="store_true",
                        help="Measure impact of vetoing entries held through earnings")
    parser.add_argument("--non-overlapping", action="store_true",
                        help="Realistic: one position per ticker at a time (no pyramiding)")
    parser.add_argument("--filter-analysis", action="store_true",
                        help="Bucket expectancy by features/flags to design entry filters")
    parser.add_argument("--reweight-test", action="store_true",
                        help="A/B the default vs candidate score weights across thresholds")
    parser.add_argument("--topn-test", action="store_true",
                        help="Test taking only the top-N highest-scoring signals per day")
    parser.add_argument("--sector-analysis", action="store_true",
                        help="Validate sector-ETF veto, sector caps, and regime-dependent exits")
    parser.add_argument("--entry-timing", action="store_true",
                        help="Compare chase-at-alert vs retrace-limit entries")
    parser.add_argument("--recent-only", action="store_true",
                        help="Use one recent ~55d window regardless of interval "
                             "(for fair 1h vs 15m comparisons)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Portfolio test vs SPY buy-and-hold and vs random selection")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Concurrent position cap for the portfolio test (default 5)")
    parser.add_argument("--benchmark-runs", type=int, default=20,
                        help="Random-selection draws per window (default 20)")
    parser.add_argument("--no-band", action="store_true",
                        help="Disable the live momentum band filter in the portfolio test")
    parser.add_argument("--exit-benchmark", action="store_true",
                        help="Rank exit strategies by portfolio return vs SPY (train-only selection)")
    args = parser.parse_args()

    if args.recent_only:
        w = _recent_15m_windows()[0]
        windows = [dict(w, name=f"recent-{args.interval}")]
    else:
        windows = WINDOWS_1H if args.interval in ('1h', '60m') else _recent_15m_windows()

    print(f"\n{'=' * 72}")
    print(f"  HARDENED BACKTEST HARNESS")
    print(f"  interval={args.interval}  threshold={args.threshold}  hold_days={args.hold_days}")
    print(f"  windows: {', '.join(w['name'] for w in windows)}")
    print(f"{'=' * 72}\n")

    print("Step 1/4: tickers + daily data")
    tickers = get_sp500_tickers()
    daily_data = load_daily(tickers, refresh=args.refresh)
    spy_close = load_spy_daily(refresh=args.refresh)

    print("\nStep 2/4: liquid subset")
    subset = pick_liquid_subset(daily_data, n=args.subset_size)
    print(f"  selected {len(subset)} most-liquid tickers: {', '.join(subset[:12])}...")

    print("\nStep 3/4: as-of daily context (lookahead-safe) + intraday data")
    provider = DailyContextProvider(daily_data, spy_close)
    intra_data = load_intraday(subset, args.interval, refresh=args.refresh)
    frames, arrays = build_frames(intra_data, args.interval)

    bars_per_day = BARS_PER_DAY.get(args.interval, 26)
    hold_bars = args.hold_days * bars_per_day

    earnings_dates = None
    if args.earnings_blackout:
        print("\n  loading earnings dates for blackout analysis...")
        earnings_dates = load_earnings(subset, refresh=args.refresh)

    adopted_exit = (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(
        args.stop_mult, args.target_mult)
    max_trades = 999 if args.non_overlapping else args.max_trades

    sector_map = sector_provider = None
    if args.sector_analysis:
        print("\n  loading sector map + sector ETF history...")
        sector_map = load_sector_map(refresh=args.refresh)
        sector_provider = SectorProvider(refresh=args.refresh)

    print(f"\nStep 4/4: collecting entries across {len(windows)} window(s)"
          f"{' (non-overlapping)' if args.non_overlapping else ''}...")
    entries = []
    for w in windows:
        ent = collect_entries(w, subset, frames, provider, args.threshold, max_trades,
                              earnings_dates=earnings_dates, hold_bars=hold_bars,
                              non_overlapping=args.non_overlapping, arrays=arrays,
                              resolve_fn=adopted_exit,
                              sector_map=sector_map, sector_provider=sector_provider)
        entries.extend(ent)
        print(f"  {w['name']:14s} -> {len(ent):4d} entries")

    if args.exit_benchmark:
        exit_benchmark(entries, arrays, spy_close, windows, hold_bars, args.slippage_bps,
                       args.regime_filtered, None if args.no_band else (-6.0, 6.0),
                       positions=(3, args.max_positions))
    elif args.benchmark:
        benchmark_test(entries, arrays, provider, subset, spy_close, windows, hold_bars,
                       args.slippage_bps, args.stop_mult, args.target_mult,
                       args.regime_filtered, args.max_positions,
                       None if args.no_band else (-6.0, 6.0), args.benchmark_runs)
    elif args.entry_timing:
        entry_timing_test(entries, arrays, hold_bars, args.slippage_bps,
                          args.stop_mult, args.target_mult, args.regime_filtered,
                          wait_bars=bars_per_day)
    elif args.sector_analysis:
        sector_regime_analysis(entries, arrays, hold_bars, args.slippage_bps,
                               args.stop_mult, args.target_mult, args.regime_filtered)
    elif args.topn_test:
        topn_test(entries, arrays, hold_bars, args.slippage_bps,
                  args.stop_mult, args.target_mult, args.regime_filtered)
    elif args.reweight_test:
        reweight_test(windows, subset, frames, arrays, provider, hold_bars, args.slippage_bps,
                      args.stop_mult, args.target_mult, args.regime_filtered,
                      max_trades, args.non_overlapping)
    elif args.filter_analysis:
        analyze_filters(entries, arrays, hold_bars, args.slippage_bps,
                        args.stop_mult, args.target_mult, args.regime_filtered)
    elif args.earnings_blackout:
        compare_earnings_blackout(entries, arrays, hold_bars, args.slippage_bps,
                                  args.stop_mult, args.target_mult, args.regime_filtered)
    elif args.sweep_thresholds:
        lo, hi = (int(x) for x in args.sweep_thresholds.split('-'))
        sweep_thresholds(windows, subset, frames, arrays, provider, lo, hi,
                         hold_bars, args.slippage_bps, args.max_trades,
                         args.stop_mult, args.target_mult, args.regime_filtered)
    elif args.compare_exits:
        compare_exits(entries, arrays, hold_bars, args.slippage_bps, args.regime_filtered)
    else:
        df = resolve_entries(
            entries, arrays,
            (lambda s, t: (lambda *a: resolve_fixed(*a, stop_mult=s, target_mult=t)))(args.stop_mult, args.target_mult),
            hold_bars, args.slippage_bps,
        )
        report(df, args.threshold, args.regime_filtered)


if __name__ == "__main__":
    main()
