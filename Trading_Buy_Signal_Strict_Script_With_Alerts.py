# file: Trading_Buy_Signal_Strict_Script_With_Alerts.py
#
# Multi-timeframe scored intraday alert system with market regime filter.
# Daily bars provide trend context (SMA200, ADX, relative strength vs SPY).
# Intraday bars provide entry timing (RSI, Bollinger Bands, volume, MACD histogram).
# Six independent signal categories replace correlated indicators for accuracy.
# A market-wide kill switch suppresses alerts when the S&P 500 is hostile.

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).with_name('.stock_screener.env'))

import argparse
import csv
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from io import StringIO
import io
import time
import json
import gzip
import smtplib
from email.message import EmailMessage
from textwrap import dedent

from myutils import get_sp500_tickers, get_nasdaq_composite_tickers

# =========================
# Global config / caches
# =========================
PRICE_DATA = {}                 # Intraday bar cache (refreshed every poll)
DAILY_DATA = {}                 # Daily bar cache (refreshed once per calendar day)
DAILY_CONTEXT = {}              # Pre-computed daily trend context per ticker
DAILY_CACHE_DATE = None         # Calendar date of last daily refresh
SPY_DAILY_CLOSE = None          # SPY daily close series for relative strength
MARKET_REGIME = None            # Market-wide regime dict (risk_on / caution / risk_off)
SECTOR_RET5 = {}                # Sector ETF -> trailing 5-day return % (daily refresh)

# GICS / yfinance sector names -> SPDR sector ETF proxy (for the sector kill switch).
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

BATCH_SIZE = 20
MIN_AVG_VOLUME_20 = 150_000
MIN_LAST_CLOSE = 3.0
SCORE_THRESHOLD = 11

lookback_days = 5               # Intraday bars for recent-signal lookback

# =========================
# Helpers
# =========================

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _extract_cols(df, ticker=None):
    """Return (close, high, low, volume) Series from single- or multi-level columns."""
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker is not None and ticker in lv0:
            sub = df[ticker]
        elif ticker is not None and ticker in lv1:
            sub = df.xs(ticker, axis=1, level=1)
        else:
            if set(['Close', 'Open', 'High', 'Low', 'Volume']).issubset(set(lv0)):
                sub = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
            else:
                sub = df.xs(df.columns.get_level_values(0)[0], axis=1, level=0)
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = sub.columns.get_level_values(0)
        close_col = sub['Close']
        high_col = sub['High']
        low_col = sub['Low']
        volume_col = sub['Volume']
    else:
        close_col = df['Close']
        high_col = df['High']
        low_col = df['Low']
        volume_col = df['Volume']
    return (
        close_col.astype(float),
        high_col.astype(float),
        low_col.astype(float),
        volume_col.astype(float),
    )


def _slice_from_batch(data, tkr):
    """Extract a single-ticker OHLCV frame from a yfinance batch download."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data.dropna().copy() if not data.empty else None

    lv0 = data.columns.get_level_values(0)
    lv1 = data.columns.get_level_values(1)
    sub = None

    if tkr in lv0:
        sub = data[tkr]
    elif tkr in lv1:
        sub = data.xs(tkr, axis=1, level=1)
    else:
        return None

    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(0)

    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in sub.columns]
    if not cols:
        return None
    sub = sub[cols].dropna(how='all')
    return sub if not sub.empty else None


def get_company_info_robust(ticker, max_retries=3):
    """Company info with retries. Returns (name, sector, market_cap)."""
    for attempt in range(max_retries):
        try:
            ti = yf.Ticker(ticker)
            company_name = ticker
            sector = "N/A"
            market_cap = np.nan

            info = {}
            try:
                info = (ti.get_info() if hasattr(ti, "get_info") else ti.info) or {}
            except Exception:
                info = {}

            name_candidate = info.get('longName') or info.get('shortName') or info.get('displayName')
            if name_candidate and str(name_candidate).strip():
                company_name = str(name_candidate).strip()

            sec_candidate = info.get('sector') or info.get('industry')
            if sec_candidate and str(sec_candidate).strip():
                sector = str(sec_candidate).strip()

            mc = info.get('marketCap')
            if mc is not None:
                try:
                    market_cap = float(mc)
                except Exception:
                    pass

            return company_name, sector, market_cap
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue

    return ticker, "N/A", np.nan


# =========================
# News & earnings filters
# =========================
# Earnings blackout is validated indirectly (expectancy-neutral but reduces
# overnight gap tail-risk the backtest can't model). The news-sentiment veto is
# a defensive, forward-validated filter: it CANNOT be backtested with free data
# (yfinance only exposes recent headlines), so its value is monitored via the
# alert CSV over time rather than proven historically.

_VADER = None

# Headlines containing these terms are treated as hard-negative regardless of
# the overall sentiment score (event risk that warrants skipping a fresh buy).
HARD_NEG_KEYWORDS = [
    'fraud', 'lawsuit', 'sued', 'investigation', 'probe', 'sec ', 'doj',
    'bankruptcy', 'chapter 11', 'delist', 'halt', 'recall', 'downgrade',
    'guidance cut', 'cuts guidance', 'slashes', 'scandal', 'default',
    'layoff', 'data breach', 'subpoena', 'short seller', 'plunge', 'warns',
]
NEWS_NEG_COMPOUND = -0.4  # average VADER compound at/below this = negative


def _get_vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def get_news_sentiment(ticker, lookback_hours=72, max_items=12):
    """Score recent headlines. Returns dict(score, label, negative, headline, n)."""
    from datetime import timezone
    result = {'score': 0.0, 'label': 'neutral', 'negative': False, 'headline': '', 'n': 0}
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return result
    if not items:
        return result

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    analyzer = _get_vader()
    scores = []
    worst_score, worst_title = 1.0, ''
    hard_neg = False

    for it in items[:max_items]:
        content = it.get('content') or {}
        title = (content.get('title') or '').strip()
        if not title:
            continue
        pub = content.get('pubDate') or content.get('displayTime')
        if pub:
            try:
                dt = datetime.fromisoformat(str(pub).replace('Z', '+00:00'))
                if dt < cutoff:
                    continue
            except Exception:
                pass
        text = f"{title}. {content.get('summary') or ''}"
        comp = analyzer.polarity_scores(text)['compound']
        scores.append(comp)
        low = text.lower()
        if any(k in low for k in HARD_NEG_KEYWORDS):
            hard_neg = True
        if comp < worst_score:
            worst_score, worst_title = comp, title

    if not scores:
        return result

    avg = sum(scores) / len(scores)
    # Veto only when the news flow is genuinely bad. A single very-negative
    # headline or a hard-negative keyword triggers ONLY when overall sentiment
    # isn't positive — otherwise one sensational/unrelated headline attached to a
    # broadly good stock would cause false vetoes (and missed good trades).
    negative = (
        (avg <= NEWS_NEG_COMPOUND)
        or (hard_neg and avg < 0.05)
        or (min(scores) <= -0.6 and avg < 0.10)
    )
    result.update({
        'score': round(avg, 3),
        'label': 'negative' if negative else ('positive' if avg > 0.2 else 'neutral'),
        'negative': bool(negative),
        'headline': worst_title[:140],
        'n': len(scores),
    })
    return result


def get_next_earnings_date(ticker, max_retries=2):
    """Nearest upcoming earnings date (date) or None."""
    for attempt in range(max_retries):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
            if ed is not None and len(ed):
                idx = ed.index
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                now = pd.Timestamp.now()
                future = [d for d in idx if d >= now]
                return min(future).date() if future else None
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.4)
                continue
    return None


def earnings_blackout_status(ticker, blackout_days):
    """(in_blackout, next_date): True if earnings falls within blackout_days ahead."""
    d = get_next_earnings_date(ticker)
    if d is None:
        return False, None
    days_until = (d - datetime.today().date()).days
    return (0 <= days_until <= blackout_days), d


# =========================
# Daily context (trend filters on DAILY bars — refreshed once per day)
# =========================

def _compute_adx(high_col, low_col, close_col, period=14):
    """Wilder's ADX. Returns (adx_series, atr_series)."""
    up_move = high_col - high_col.shift(1)
    down_move = low_col.shift(1) - low_col

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close_col.shift(1)
    tr = pd.concat([
        high_col - low_col,
        (high_col - prev_close).abs(),
        (low_col - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * smooth_plus / atr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus / atr.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, atr


def compute_market_regime(spy_close):
    """
    Evaluate S&P 500 (SPY) health to determine market regime.
    Uses SMA50/SMA200 position, recent drawdown, and market breadth.

    Returns dict:
        regime: 'risk_on' | 'caution' | 'risk_off'
        threshold_adjust: int added to score threshold in caution mode
        spy_above_sma50, spy_above_sma200, spy_5d_return, breadth_pct
    """
    result = {
        'regime': 'risk_on',
        'threshold_adjust': 0,
        'spy_above_sma50': True,
        'spy_above_sma200': True,
        'spy_5d_return': 0.0,
        'breadth_pct': 100.0,
    }

    if spy_close is None or len(spy_close) < 200:
        return result

    latest = float(spy_close.iloc[-1])
    sma50 = float(spy_close.rolling(50).mean().iloc[-1])
    sma200 = float(spy_close.rolling(200).mean().iloc[-1])
    ret_5d = float((spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100) if len(spy_close) >= 6 else 0.0

    result['spy_above_sma50'] = latest > sma50
    result['spy_above_sma200'] = latest > sma200
    result['spy_5d_return'] = ret_5d

    # Breadth: % of already-computed tickers above their daily SMA200
    if DAILY_CONTEXT:
        n_above = sum(1 for v in DAILY_CONTEXT.values() if v.get('above_sma200'))
        result['breadth_pct'] = (n_above / len(DAILY_CONTEXT)) * 100.0

    # --- Regime classification ---
    # risk_off: severe conditions — suppress all alerts
    if (not result['spy_above_sma200']) or ret_5d < -5.0:
        result['regime'] = 'risk_off'
        result['threshold_adjust'] = 99
    # caution: deteriorating conditions — raise threshold
    elif (not result['spy_above_sma50']) or result['breadth_pct'] < 40.0 or ret_5d < -2.0:
        result['regime'] = 'caution'
        result['threshold_adjust'] = 3
    else:
        result['regime'] = 'risk_on'
        result['threshold_adjust'] = 0

    return result


def refresh_daily_context(tickers, batch_size=20):
    """
    Download daily bars and compute per-ticker trend context:
    SMA200, ADX, ATR, and 20-day relative strength vs SPY.
    Also computes market regime from SPY data.
    Cached for the calendar day — skipped if already fresh.
    """
    global DAILY_DATA, DAILY_CONTEXT, DAILY_CACHE_DATE, SPY_DAILY_CLOSE, MARKET_REGIME, SECTOR_RET5

    today = datetime.today().date()
    if DAILY_CACHE_DATE == today and DAILY_CONTEXT:
        return

    print("📅 Refreshing daily trend context (runs once per trading day)...")
    DAILY_DATA.clear()
    DAILY_CONTEXT.clear()

    daily_start = datetime.today() - timedelta(days=400)
    daily_end = datetime.today()

    # SPY for relative-strength benchmark and market regime
    try:
        spy_raw = yf.download("SPY", start=daily_start, end=daily_end, progress=False)
        if spy_raw is not None and not spy_raw.empty:
            spy_close, _, _, _ = _extract_cols(spy_raw, ticker="SPY")
            SPY_DAILY_CLOSE = spy_close.astype(float)
        else:
            SPY_DAILY_CLOSE = None
    except Exception as e:
        print(f"  ⚠️  Failed to download SPY daily data: {e}")
        SPY_DAILY_CLOSE = None

    # Sector ETFs for the sector-level kill switch (trailing 5-day return each)
    SECTOR_RET5 = {}
    try:
        etfs = sorted(set(SECTOR_ETF.values()))
        etf_raw = yf.download(etfs, start=daily_start, end=daily_end,
                              progress=False, group_by='ticker', threads=True)
        if etf_raw is not None and not etf_raw.empty:
            for etf in etfs:
                try:
                    sub = _slice_from_batch(etf_raw, etf)
                    if sub is None or sub.empty:
                        continue
                    close_e, _, _, _ = _extract_cols(sub, ticker=etf)
                    r5 = close_e.pct_change(5).iloc[-1]
                    if pd.notna(r5):
                        SECTOR_RET5[etf] = float(r5 * 100.0)
                except Exception:
                    continue
        weak = {k: v for k, v in SECTOR_RET5.items() if v <= -5.0}
        if weak:
            print(f"  🔻 Weak sectors (5d <= -5%): "
                  + ", ".join(f"{k} {v:+.1f}%" for k, v in sorted(weak.items(), key=lambda x: x[1])))
    except Exception as e:
        print(f"  ⚠️  Failed to download sector ETF data: {e}")

    for batch in _chunks(tickers, batch_size):
        try:
            data = yf.download(
                batch, start=daily_start, end=daily_end,
                progress=False, group_by='ticker', threads=True,
            )
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    for tkr in batch:
                        try:
                            sub = _slice_from_batch(data, tkr)
                            if sub is not None and not sub.empty:
                                DAILY_DATA[tkr] = sub
                        except Exception:
                            pass
                elif len(batch) == 1:
                    DAILY_DATA[batch[0]] = data.dropna().copy()
        except Exception as e:
            print(f"  ⚠️  Daily batch download error: {e}")

    for tkr, df in DAILY_DATA.items():
        try:
            if len(df) < 220:
                continue

            close_col, high_col, low_col, volume_col = _extract_cols(df, ticker=tkr)

            # -- SMA200 --
            sma200 = close_col.rolling(200, min_periods=200).mean()
            latest_close = float(close_col.iloc[-1])
            sma200_val = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else np.nan
            above_sma200 = bool(pd.notna(sma200_val) and latest_close > sma200_val)

            # -- ADX & ATR on daily --
            adx_series, atr_series = _compute_adx(high_col, low_col, close_col, period=14)
            adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
            daily_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else np.nan

            # -- 20-day relative strength vs SPY --
            rs_outperforming = False
            if SPY_DAILY_CLOSE is not None and len(SPY_DAILY_CLOSE) > 20:
                common_idx = close_col.index.intersection(SPY_DAILY_CLOSE.index)
                if len(common_idx) > 20:
                    tkr_ret = close_col.loc[common_idx].pct_change(20).iloc[-1]
                    spy_ret = SPY_DAILY_CLOSE.loc[common_idx].pct_change(20).iloc[-1]
                    if pd.notna(tkr_ret) and pd.notna(spy_ret):
                        rs_outperforming = bool(tkr_ret > spy_ret)

            # -- 5-day return (momentum band / falling-knife guard) --
            ret_5d = 0.0
            if len(close_col) >= 6:
                r5 = close_col.pct_change(5).iloc[-1]
                if pd.notna(r5):
                    ret_5d = float(r5 * 100.0)

            DAILY_CONTEXT[tkr] = {
                'above_sma200': above_sma200,
                'sma200': sma200_val,
                'daily_close': latest_close,
                'adx': adx_val,
                'adx_trending': adx_val > 25,
                'adx_strong': adx_val > 35,
                'daily_atr': daily_atr,
                'rs_outperforming': rs_outperforming,
                'ret_5d': ret_5d,
            }
        except Exception:
            continue

    # Compute market regime now that daily context is populated
    MARKET_REGIME = compute_market_regime(SPY_DAILY_CLOSE)
    regime = MARKET_REGIME['regime']
    breadth = MARKET_REGIME['breadth_pct']
    spy_ret = MARKET_REGIME['spy_5d_return']

    regime_emoji = {'risk_on': '🟢', 'caution': '🟡', 'risk_off': '🔴'}
    print(
        f"  {regime_emoji.get(regime, '⚪')} Market regime: {regime.upper()}"
        f"  (SPY 5d: {spy_ret:+.1f}%, breadth: {breadth:.0f}%"
        f"  SMA50: {'above' if MARKET_REGIME['spy_above_sma50'] else 'BELOW'}"
        f"  SMA200: {'above' if MARKET_REGIME['spy_above_sma200'] else 'BELOW'})"
    )

    DAILY_CACHE_DATE = today
    print(f"  ✅ Daily context ready for {len(DAILY_CONTEXT)} tickers")


# =========================
# Intraday indicators (computed on intraday bars for entry timing)
# =========================

def calculate_intraday_indicators(ticker):
    """
    Compute intraday entry-timing indicators across 4 independent categories:
      1. Momentum  — RSI (Wilder) with directional confirmation
      2. Mean reversion — Bollinger Bands bounce / cross
      3. Volume   — participation vs 20-bar average
      4. Momentum acceleration — MACD histogram direction

    EMA5/EMA20 are kept as display values but NOT scored independently
    (they're correlated with MACD). SMA20/50 crossover is removed entirely.
    """
    try:
        if ticker not in PRICE_DATA:
            return None

        df = PRICE_DATA[ticker].copy()
        if df is None or df.empty or len(df) < 40:
            return None

        close, high, low, volume = _extract_cols(df, ticker=ticker)

        # -- EMA (display only, not scored independently) --
        df["EMA5"] = close.ewm(span=5, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()

        # -- RSI (Wilder's exponential smoothing) with rising check --
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
        df["RSI_Rising"] = df["RSI"] > df["RSI"].shift(3)

        # -- Bollinger Bands --
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

        # -- Volume --
        df["Volume_MA"] = volume.rolling(20, min_periods=20).mean()
        df["Volume_Above_Avg"] = volume > df["Volume_MA"]
        df["Volume_Surge"] = volume > df["Volume_MA"] * 1.5

        # -- ATR (intraday, Wilder's) --
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR14"] = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

        # -- MACD (intraday 12-26-9) — histogram only for scoring --
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD_Line"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
        df["MACD_Hist_Positive"] = df["MACD_Hist"] > 0
        df["MACD_Hist_Rising"] = df["MACD_Hist"] > df["MACD_Hist"].shift(1)

        # -- BB position (0% = lower band, 100% = upper band) --
        df["BB_Position"] = ((close - df["BB_Lower"]) / df["BB_Range"]) * 100.0

        df = df.dropna(subset=["RSI", "EMA5", "EMA20", "BB_Middle"])
        return df if not df.empty else None

    except Exception as e:
        print(f"  Error computing intraday indicators for {ticker}: {e}")
        return None


# =========================
# Scoring system — 6 independent signal categories
# =========================
#
# Each category captures a distinct market dimension to avoid double-counting
# correlated moving-average signals.
#
# DAILY CONTEXT (max 7):
#   1. Trend      — Above SMA200               +3   (strongest single predictor)
#   2. Trend Qual — ADX >35: +2, ADX >25: +1         (trending vs ranging)
#   3. Rel Strength — Outperforming SPY 20d     +2   (stock selection alpha)
#
# INTRADAY ENTRY (max 10):
#   4. Momentum   — RSI 40-60 + rising: +3, RSI 60-70: +1  (not overbought)
#   5. Mean Revert — BB lower bounce: +3, BB mid cross: +1
#   6. Volume     — Surge >1.5x: +2, above avg: +1
#   7. Momentum Accel — MACD hist >0 & rising: +2, hist rising: +1
#
# Grand max = 17      Default threshold = 9 (~53%)

def compute_signal_score(intra_df, daily_ctx, lookback_bars=5):
    """
    Combine daily trend context with intraday entry signals into a numeric
    score using 6 independent categories.
    Returns (score, details_dict) or (0, None).
    """
    if intra_df is None or intra_df.empty or daily_ctx is None:
        return 0, None

    N = max(3, int(lookback_bars))
    recent = intra_df.tail(N)
    if recent.empty:
        return 0, None

    score = 0
    details = {}
    now = intra_df.iloc[-1]

    # ===== 1. Trend — SMA200 (daily) =====
    if daily_ctx.get('above_sma200', False):
        score += 3
        details['above_sma200'] = True
    else:
        details['above_sma200'] = False

    # ===== 2. Trend quality — ADX (daily) =====
    adx_val = daily_ctx.get('adx', 0.0)
    details['adx'] = adx_val
    if daily_ctx.get('adx_strong', False):
        score += 2
        details['adx_tier'] = 'strong'
    elif daily_ctx.get('adx_trending', False):
        score += 1
        details['adx_tier'] = 'trending'
    else:
        details['adx_tier'] = 'weak'

    # ===== 3. Relative strength vs SPY (daily) =====
    if daily_ctx.get('rs_outperforming', False):
        score += 2
        details['rs_outperforming'] = True
    else:
        details['rs_outperforming'] = False

    # ===== 4. Momentum — RSI (intraday) =====
    rsi_val = float(now["RSI"]) if pd.notna(now.get("RSI")) else np.nan
    rsi_rising = bool(now.get("RSI_Rising")) if pd.notna(now.get("RSI_Rising")) else False
    details['rsi'] = rsi_val
    details['rsi_rising'] = rsi_rising

    if pd.notna(rsi_val):
        if 40 <= rsi_val <= 60 and rsi_rising:
            score += 3
            details['rsi_zone'] = 'sweet_spot'
        elif 60 < rsi_val <= 70:
            score += 1
            details['rsi_zone'] = 'warm'
        else:
            details['rsi_zone'] = 'neutral'
    else:
        details['rsi_zone'] = 'na'

    # ===== 5. Mean reversion — Bollinger Bands (intraday) =====
    if "BB_Lower_Bounce" in recent.columns and recent["BB_Lower_Bounce"].any():
        score += 3
        details['bb_lower_bounce'] = True
    elif "BB_Buy" in recent.columns and recent["BB_Buy"].any():
        score += 1
        details['bb_middle_cross'] = True

    # ===== 6. Volume confirmation (intraday) =====
    if pd.notna(now.get("Volume_Surge")) and bool(now["Volume_Surge"]):
        score += 2
        details['volume_surge'] = True
    elif pd.notna(now.get("Volume_Above_Avg")) and bool(now["Volume_Above_Avg"]):
        score += 1
        details['volume_above_avg'] = True

    # ===== 7. Momentum acceleration — MACD histogram (intraday) =====
    hist_pos = bool(now.get("MACD_Hist_Positive")) if pd.notna(now.get("MACD_Hist_Positive")) else False
    hist_rising = bool(now.get("MACD_Hist_Rising")) if pd.notna(now.get("MACD_Hist_Rising")) else False

    if hist_pos and hist_rising:
        score += 2
        details['macd_hist'] = 'strong'
    elif hist_rising:
        score += 1
        details['macd_hist'] = 'improving'
    else:
        details['macd_hist'] = 'neutral'

    # ===== Risk / reward context (daily ATR) =====

    price = float(now["Close"]) if pd.notna(now.get("Close")) else np.nan
    daily_atr = daily_ctx.get('daily_atr', np.nan)

    if pd.notna(price) and pd.notna(daily_atr) and daily_atr > 0:
        # Primary stop/target = 1.0x / 3.0x ATR (best expectancy in multi-period
        # backtest; see backtest_harness.py exit comparison). 1.5x/2x kept secondary.
        stop_1x = price - 1.0 * daily_atr
        stop_1_5x = price - 1.5 * daily_atr
        stop_2x = price - 2.0 * daily_atr
        target_2x = price + 2.0 * daily_atr
        target_3x = price + 3.0 * daily_atr
        atr_pct = (daily_atr / price) * 100.0
        risk = price - stop_1x
        rr_ratio = (target_3x - price) / risk if risk > 0 else 0.0
    else:
        stop_1x = stop_1_5x = stop_2x = target_2x = target_3x = np.nan
        atr_pct = rr_ratio = np.nan

    ema5 = float(now["EMA5"]) if pd.notna(now.get("EMA5")) else np.nan
    ema20 = float(now["EMA20"]) if pd.notna(now.get("EMA20")) else np.nan
    bb_pos = float(now["BB_Position"]) if pd.notna(now.get("BB_Position")) else np.nan

    details.update({
        'price': price,
        'ema5': ema5,
        'ema20': ema20,
        'bb_position': bb_pos,
        'daily_atr': daily_atr,
        'atr_pct': atr_pct,
        'stop_1x': stop_1x,
        'stop_1_5x': stop_1_5x,
        'stop_2x': stop_2x,
        'target_2x': target_2x,
        'target_3x': target_3x,
        'rr_ratio': rr_ratio,
        'sma200_daily': daily_ctx.get('sma200', np.nan),
    })

    return score, details


# =========================
# Email
# =========================

def send_email_alert(smtp_host, smtp_port, smtp_user, smtp_pass, to_list, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user if smtp_user else "alerts@localhost"
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            try:
                s.starttls()
                s.ehlo()
            except Exception:
                pass
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)


# =========================
# Alert CSV log (for later win-rate analysis)
# =========================

_CSV_HEADER = [
    'timestamp', 'ticker', 'company', 'sector', 'score', 'market_regime',
    'price', 'rsi', 'rsi_zone', 'daily_atr', 'atr_pct',
    'stop_1x', 'stop_1_5x', 'stop_2x', 'target_2x', 'target_3x', 'rr_ratio',
    'above_sma200', 'adx', 'adx_tier', 'rs_outperforming',
    'volume_surge', 'bb_lower_bounce', 'macd_hist',
    'stk_ret_5d', 'sector_ret_5d', 'next_earnings', 'news_sentiment', 'news_label',
    'news_headline', 'veto_reason',
]


def log_alert_to_csv(log_path, ticker, score, details, company_name, sector):
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_CSV_HEADER)

        def _fmt(val, decimals=2):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ''
            return f"{val:.{decimals}f}" if isinstance(val, float) else str(val)

        regime = MARKET_REGIME['regime'] if MARKET_REGIME else ''
        writer.writerow([
            datetime.now().isoformat(),
            ticker,
            company_name,
            sector,
            score,
            regime,
            _fmt(details.get('price'), 2),
            _fmt(details.get('rsi'), 1),
            details.get('rsi_zone', ''),
            _fmt(details.get('daily_atr'), 2),
            _fmt(details.get('atr_pct'), 2),
            _fmt(details.get('stop_1x'), 2),
            _fmt(details.get('stop_1_5x'), 2),
            _fmt(details.get('stop_2x'), 2),
            _fmt(details.get('target_2x'), 2),
            _fmt(details.get('target_3x'), 2),
            _fmt(details.get('rr_ratio'), 1),
            details.get('above_sma200', ''),
            _fmt(details.get('adx'), 1),
            details.get('adx_tier', ''),
            details.get('rs_outperforming', ''),
            details.get('volume_surge', ''),
            details.get('bb_lower_bounce', ''),
            details.get('macd_hist', ''),
            _fmt(details.get('stk_ret_5d'), 2),
            _fmt(details.get('sector_ret_5d'), 2),
            details.get('next_earnings', ''),
            _fmt(details.get('news_sentiment'), 3),
            details.get('news_label', ''),
            details.get('news_headline', ''),
            details.get('veto_reason', ''),
        ])


# =========================
# Market hours (DST-aware via zoneinfo)
# =========================

def _market_open_now():
    from datetime import time as dt_time
    PT = ZoneInfo("America/Los_Angeles")
    now = datetime.now(PT)
    if now.weekday() >= 5:
        return False
    return dt_time(6, 30) <= now.time() <= dt_time(13, 0)


def _parse_hhmm(s):
    from datetime import time as dt_time
    hh, mm = s.strip().split(":")
    return dt_time(int(hh), int(mm))


def build_daily_picks(log_path, min_score, max_picks):
    """Rank today's logged (non-vetoed) signals using pick_daily_alerts logic.
    Returns (picks_df, alternates_df) or (None, None) if nothing to rank yet."""
    from pick_daily_alerts import rank_alerts
    if not os.path.exists(log_path):
        return None, None
    df = pd.read_csv(log_path)
    day = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
    df = df[df['timestamp'].str.startswith(day)]
    df = df[df['veto_reason'].fillna('') == '']
    df = df[pd.to_numeric(df['score'], errors='coerce') >= min_score]
    if df.empty:
        return None, None
    return rank_alerts(df, max_picks=max_picks)


def _format_picks_body(picks, alternates, max_picks):
    lines = [f"Top {len(picks)} picks (one per sector, mild-dip & volume-surge first):", ""]
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        why = []
        r5 = r.get('stk_ret_5d')
        if pd.notna(r5):
            why.append(f"5d {float(r5):+.1f}%" + (" (mild dip)" if r.get('mild_dip') else ""))
        if r.get('vol_surge'):
            why.append("volume surge")
        why.append(f"score {float(r['score']):.0f}")
        lines.append(f"{i}. {r['ticker']:6s} ${float(r['price']):.2f}  "
                     f"{str(r.get('sector', '?'))}  [{', '.join(why)}]")
        lines.append(f"   stop ${float(r['stop_1x']):.2f}  |  target ${float(r['target_3x']):.2f}")
        lines.append("")
    if alternates is not None and len(alternates):
        alt = ", ".join(f"{r.ticker}({float(r.score):.0f})"
                        for r in alternates.head(8).itertuples())
        lines.append(f"Alternates: {alt}")
    lines.append("")
    lines.append("Individual alert emails start now. Full pool: alert_log.csv")
    return "\n".join(lines)


def _price_sparkline(tkr, days=20):
    """Unicode sparkline of the last `days` daily closes, e.g. '▃▄▆█▇▅ 20d -3.2%'.
    Display-only nice-to-have: returns '' on any problem so alerts never break."""
    try:
        df = DAILY_DATA.get(tkr)
        if df is None or df.empty:
            return ""
        close_col, _, _, _ = _extract_cols(df, ticker=tkr)
        closes = close_col.dropna().tail(days)
        if len(closes) < 5:
            return ""
        vals = closes.to_numpy(dtype=float)
        lo, hi = vals.min(), vals.max()
        blocks = "▁▂▃▄▅▆▇█"
        if hi - lo < 1e-9:
            spark = blocks[3] * len(vals)
        else:
            spark = "".join(blocks[int((v - lo) / (hi - lo) * 7)] for v in vals)
        pct = (vals[-1] / vals[0] - 1) * 100.0
        return f"{spark}  {len(vals)}d {pct:+.1f}%"
    except Exception:
        return ""


# =========================
# Build alert email body
# =========================

def _build_alert_body(tkr, company_name, sector, score, details, interval, threshold):
    price = details.get('price', 0)
    ema5 = details.get('ema5', 0)
    ema20 = details.get('ema20', 1)
    ema_spread = ((ema5 - ema20) / max(abs(ema20), 0.01)) * 100.0

    regime_str = "N/A"
    if MARKET_REGIME:
        r = MARKET_REGIME
        regime_str = (
            f"{r['regime'].upper()}  "
            f"(SPY 5d: {r['spy_5d_return']:+.1f}%, breadth: {r['breadth_pct']:.0f}%)"
        )

    checks = []

    # --- Daily context ---
    checks.append("  --- Daily Trend Context ---")
    if details.get('above_sma200'):
        checks.append("  [+3] Above daily SMA200")
    else:
        checks.append("  [--] Below daily SMA200")

    adx_tier = details.get('adx_tier', 'weak')
    adx_val = details.get('adx', 0)
    if adx_tier == 'strong':
        checks.append(f"  [+2] ADX strong trend ({adx_val:.1f})")
    elif adx_tier == 'trending':
        checks.append(f"  [+1] ADX trending ({adx_val:.1f})")
    else:
        checks.append(f"  [--] ADX weak / ranging ({adx_val:.1f})")

    if details.get('rs_outperforming'):
        checks.append("  [+2] Outperforming SPY (20-day)")
    else:
        checks.append("  [--] Underperforming SPY (20-day)")

    # --- Intraday entry ---
    checks.append("")
    checks.append("  --- Intraday Entry Signals ---")

    rsi_zone = details.get('rsi_zone', 'na')
    rsi_val = details.get('rsi', 0)
    rsi_dir = "rising" if details.get('rsi_rising') else "flat/falling"
    if rsi_zone == 'sweet_spot':
        checks.append(f"  [+3] RSI {rsi_val:.1f} in sweet spot (40-60, {rsi_dir})")
    elif rsi_zone == 'warm':
        checks.append(f"  [+1] RSI {rsi_val:.1f} getting warm (60-70)")
    else:
        checks.append(f"  [--] RSI {rsi_val:.1f} ({rsi_dir})")

    if details.get('bb_lower_bounce'):
        checks.append("  [+3] Bollinger lower-band bounce")
    elif details.get('bb_middle_cross'):
        checks.append("  [+1] Bollinger middle-band cross")
    else:
        checks.append("  [--] No Bollinger signal")

    if details.get('volume_surge'):
        checks.append("  [+2] Volume surge (>1.5x 20-day avg)")
    elif details.get('volume_above_avg'):
        checks.append("  [+1] Volume above 20-day average")
    else:
        checks.append("  [--] Volume below average")

    macd_hist = details.get('macd_hist', 'neutral')
    if macd_hist == 'strong':
        checks.append("  [+2] MACD histogram positive & accelerating")
    elif macd_hist == 'improving':
        checks.append("  [+1] MACD histogram improving")
    else:
        checks.append("  [--] MACD histogram flat/declining")

    checks_str = "\n".join(checks)

    def _safe(val, fmt=".2f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt}}"

    return dedent(f"""\
        Multi-timeframe scored BUY signal detected!

        Ticker:       {tkr}
        Company:      {company_name}
        Sector:       {sector}
        Score:        {score} / 17  (threshold: {threshold})
        Interval:     {interval}
        Market:       {regime_str}

        === Price & Momentum ===
        Trend (4 weeks):  {_price_sparkline(tkr) or 'n/a'}
        Price:            ${_safe(price)}
        RSI (Wilder 14):  {_safe(details.get('rsi'), '.1f')}
        EMA 5/20 spread:  {ema_spread:.2f}%
        BB position:      {_safe(details.get('bb_position'), '.1f')}%

        === Risk Management (based on daily ATR) ===
        Daily ATR:        ${_safe(details.get('daily_atr'))}  ({_safe(details.get('atr_pct'), '.1f')}% of price)
        > Stop  1.0x:     ${_safe(details.get('stop_1x'))}   (recommended)
        > Target 3.0x:    ${_safe(details.get('target_3x'))}   (recommended)
        > R:R (3x/1x):    {_safe(details.get('rr_ratio'), '.1f')} : 1
        secondary stops:  1.5x ${_safe(details.get('stop_1_5x'))} | 2.0x ${_safe(details.get('stop_2x'))}
        secondary target: 2.0x ${_safe(details.get('target_2x'))}

        === News & Events ===
        5-day move:       {_safe(details.get('stk_ret_5d'), '+.1f')}%  (band-checked)
        Sector ETF 5d:    {_safe(details.get('sector_ret_5d'), '+.1f')}%  (guard-checked)
        Next earnings:    {details.get('next_earnings') or 'unknown'}
        News sentiment:   {details.get('news_label', 'n/a')} ({_safe(details.get('news_sentiment'), '+.2f') if details.get('news_sentiment') is not None else 'n/a'})
        Latest headline:  {details.get('news_headline') or '(none in window)'}

        === Signal Checklist (6 independent categories) ===
{checks_str}

        This alert fires when the score crosses {threshold} from below.
    """)


# =========================
# CLI entrypoint — live intraday alert loop
# =========================

def run_cli(args):
    global MIN_AVG_VOLUME_20, MIN_LAST_CLOSE, lookback_days, BATCH_SIZE, SCORE_THRESHOLD

    MIN_AVG_VOLUME_20 = args.min_volume
    MIN_LAST_CLOSE = args.min_price
    lookback_days = args.lookback
    BATCH_SIZE = args.batch
    SCORE_THRESHOLD = args.score_threshold

    if not args.live:
        print("This script is for intraday alerts only. Use --live flag.")
        return

    smtp_user = args.smtp_user or os.environ.get("ALERT_FROM_EMAIL", "")
    to_list_raw = args.email_to or os.environ.get("ALERT_TO_EMAILS", "")
    to_list = [e.strip() for e in to_list_raw.split(",") if e.strip()]

    if not to_list:
        print("--live requires recipients. Provide --email-to or set ALERT_TO_EMAILS in .stock_screener.env")
        return

    smtp_pass = os.environ.get(args.smtp_pass_env, "")

    print(f"🔄 Live intraday alerts: interval={args.interval}, period={args.period}, poll every {args.poll_secs}s")
    print(f"📧 Email to: {to_list} via {args.smtp_host}:{args.smtp_port} as {smtp_user or '(no user)'}")
    print(f"🎯 Score threshold: {SCORE_THRESHOLD} / 17, debounce: {args.debounce_mins}min")
    print(f"🛡️  Market kill switch: risk_off suppresses all; caution raises threshold by +3")
    print(f"📰 Earnings blackout: {'ON' if args.earnings_blackout else 'OFF'} "
          f"(<= {args.earnings_blackout_days}d)  |  News veto: {'ON' if args.news_veto else 'OFF'} "
          f"(last {args.news_lookback_hours}h)")
    print(f"🔪 Momentum band: {'ON' if args.momentum_band else 'OFF'} "
          f"(veto if 5d move outside {args.band_min:+.0f}%..{args.band_max:+.0f}%)")
    print(f"🏭 Sector guard: {'ON' if args.sector_guard else 'OFF'} "
          f"(veto if sector ETF 5d <= {args.sector_min:+.0f}%)  |  "
          f"Caution regime: {'SKIP ALL' if args.skip_caution else 'raise threshold'}")
    print(f"📧 Email tier: {'mild-dip only' if args.email_mild_dip_only else 'all signals'}, "
          f"max {args.max_emails_per_day}/day, {args.max_emails_per_sector}/sector/day "
          f"(all signals still logged to CSV)")
    if args.daily_picks:
        print(f"📋 Daily picks: quiet until {args.picks_time} PT, then top-{args.picks_count} "
              f"picks email; individual alerts resume after")

    LAST_SCORE = {}
    LAST_ALERT_TS = {}

    # Daily email budget (reset each calendar day). Everything above threshold
    # is still logged to CSV for pick_daily_alerts.py; only the highest-quality
    # signals consume the email budget.
    email_day = None
    emails_today = 0
    emails_by_sector = {}
    log_path = args.alert_log

    # Daily picks: quiet period until picks_time, then one ranked summary email.
    picks_time = _parse_hhmm(args.picks_time)
    picks_sent_day = None

    if args.universe.lower() == "nasdaq":
        tickers = get_nasdaq_composite_tickers()
    else:
        tickers = get_sp500_tickers()

    print(f"📊 Monitoring {len(tickers)} tickers...")

    try:
        while True:
            if _market_open_now():
                print(f"\n🔍 Scanning at {datetime.now().strftime('%H:%M:%S')}...")

                # Step 1: refresh daily context (once per calendar day)
                refresh_daily_context(tickers, batch_size=BATCH_SIZE)

                # Step 2: download intraday bars
                PRICE_DATA.clear()
                for batch in _chunks(tickers, BATCH_SIZE):
                    try:
                        data = yf.download(
                            batch, period=args.period, interval=args.interval,
                            progress=False, group_by='ticker', threads=True,
                        )
                        if data is not None and not data.empty:
                            if isinstance(data.columns, pd.MultiIndex):
                                for tkr in batch:
                                    try:
                                        sub = _slice_from_batch(data, tkr)
                                        if sub is not None and not sub.empty:
                                            PRICE_DATA[tkr] = sub
                                    except Exception:
                                        pass
                            elif len(batch) == 1:
                                PRICE_DATA[batch[0]] = data.dropna().copy()
                    except Exception as e:
                        print(f"  ❌ Intraday batch error: {e}")
                        continue

                # Step 2b: check market regime kill switch
                regime = MARKET_REGIME['regime'] if MARKET_REGIME else 'risk_on'
                effective_threshold = SCORE_THRESHOLD + (MARKET_REGIME['threshold_adjust'] if MARKET_REGIME else 0)

                if regime == 'risk_off':
                    print("  🔴 RISK OFF — alerts suppressed (SPY below SMA200 or sharp drawdown)")
                    time.sleep(max(15, int(args.poll_secs)))
                    continue

                if regime == 'caution':
                    if args.skip_caution:
                        # Backtest: caution-regime entries lost money even at the
                        # raised threshold (-0.11R vs +0.19R in risk_on). Sitting
                        # out beats trading them.
                        print("  🟡 CAUTION — alerts suppressed (validated: caution trades are -EV)")
                        time.sleep(max(15, int(args.poll_secs)))
                        continue
                    print(f"  🟡 CAUTION — threshold raised to {effective_threshold} (from {SCORE_THRESHOLD})")

                # Daily-picks flow: before picks_time, individual alert emails
                # are held (signals still scanned + logged). At picks_time, one
                # ranked "DAILY PICKS" email goes out; then live emails resume.
                quiet_period = False
                if args.daily_picks:
                    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
                    if now_pt.time() < picks_time:
                        quiet_period = True
                    elif picks_sent_day != now_pt.date():
                        picks, alts = build_daily_picks(log_path, SCORE_THRESHOLD, args.picks_count)
                        if picks is None or not len(picks):
                            print(f"  📋 Picks due ({args.picks_time}) but no candidates logged yet — retrying next scan")
                        else:
                            subj = (f"DAILY PICKS {now_pt.strftime('%Y-%m-%d')}: "
                                    + ", ".join(picks['ticker'].tolist()))
                            body = _format_picks_body(picks, alts, args.picks_count)
                            try:
                                send_email_alert(
                                    args.smtp_host, args.smtp_port,
                                    smtp_user=smtp_user, smtp_pass=smtp_pass,
                                    to_list=to_list, subject=subj, body=body,
                                )
                                picks_sent_day = now_pt.date()
                                print(f"  📋 DAILY PICKS sent: {', '.join(picks['ticker'].tolist())}")
                            except Exception as ee:
                                print(f"  ⚠️  Daily picks email failed: {ee}")

                now_ts = time.time()
                alerts_sent = 0
                scored_list = []

                # Step 3: score each ticker
                for tkr in tickers:
                    try:
                        if tkr not in PRICE_DATA:
                            continue

                        df_raw = PRICE_DATA[tkr]
                        if len(df_raw) < 20:
                            continue

                        vol_ok = (
                            MIN_AVG_VOLUME_20 <= 0 or
                            float(df_raw['Volume'].tail(20).mean()) >= float(MIN_AVG_VOLUME_20)
                        )
                        px_ok = (
                            MIN_LAST_CLOSE <= 0 or
                            float(df_raw['Close'].iloc[-1]) >= float(MIN_LAST_CLOSE)
                        )
                        if not (vol_ok and px_ok):
                            continue

                        intra_df = calculate_intraday_indicators(tkr)
                        if intra_df is None or intra_df.empty:
                            continue

                        daily_ctx = DAILY_CONTEXT.get(tkr)
                        if daily_ctx is None:
                            continue

                        score, details = compute_signal_score(
                            intra_df, daily_ctx, lookback_bars=lookback_days,
                        )

                        prev_score = LAST_SCORE.get(tkr, 0)
                        crossed_threshold = (prev_score < effective_threshold) and (score >= effective_threshold)

                        if score >= effective_threshold:
                            scored_list.append((tkr, score, details))

                        if crossed_threshold:
                            last_sent = LAST_ALERT_TS.get(tkr, 0)
                            debounce_secs = args.debounce_mins * 60
                            if now_ts - last_sent >= debounce_secs:
                                price = details.get('price', 0)

                                try:
                                    company_name, sector, _ = get_company_info_robust(tkr, max_retries=1)
                                except Exception:
                                    company_name, sector = tkr, "N/A"

                                # --- Event/news vetoes (evaluated only for genuine candidates) ---
                                veto_reason = ""

                                # Momentum band (falling-knife / chase guard). Backtest
                                # validated: expectancy concentrates in mild dips; 5d
                                # moves beyond -6%/+6% underperform badly out-of-sample.
                                if args.momentum_band:
                                    ret_5d = daily_ctx.get('ret_5d', 0.0)
                                    details['stk_ret_5d'] = ret_5d
                                    if not (args.band_min < ret_5d < args.band_max):
                                        direction = "falling knife" if ret_5d <= args.band_min else "over-extended"
                                        veto_reason = f"5d move {ret_5d:+.1f}% outside band ({direction})"

                                # Sector kill switch: skip buys whose sector ETF is in
                                # freefall (backtest: sector 5d <= -5% trades were -0.29R).
                                if not veto_reason and args.sector_guard:
                                    etf = SECTOR_ETF.get(str(sector).strip())
                                    sec_r5 = SECTOR_RET5.get(etf) if etf else None
                                    if sec_r5 is not None:
                                        details['sector_ret_5d'] = sec_r5
                                        if sec_r5 <= args.sector_min:
                                            veto_reason = f"sector crash ({etf} 5d {sec_r5:+.1f}%)"

                                if not veto_reason and args.earnings_blackout:
                                    try:
                                        in_blackout, next_earn = earnings_blackout_status(
                                            tkr, args.earnings_blackout_days)
                                    except Exception:
                                        in_blackout, next_earn = False, None
                                    details['next_earnings'] = next_earn.isoformat() if next_earn else ''
                                    if in_blackout:
                                        veto_reason = f"earnings {next_earn} (<= {args.earnings_blackout_days}d)"

                                if not veto_reason and args.news_veto:
                                    try:
                                        senti = get_news_sentiment(tkr, lookback_hours=args.news_lookback_hours)
                                    except Exception:
                                        senti = {'score': 0.0, 'label': 'na', 'negative': False, 'headline': '', 'n': 0}
                                    details['news_sentiment'] = senti['score']
                                    details['news_label'] = senti['label']
                                    details['news_headline'] = senti['headline']
                                    if senti['negative']:
                                        veto_reason = f"negative news ({senti['score']:+.2f}): {senti['headline'][:60]}"
                                details['veto_reason'] = veto_reason

                                if veto_reason:
                                    LAST_ALERT_TS[tkr] = now_ts  # debounce re-evaluation
                                    print(f"  🚫 VETOED [{score}pts]: {tkr} — {veto_reason}")
                                    log_alert_to_csv(log_path, tkr, score, details, company_name, sector)
                                else:
                                    # --- Email quality tier + daily budget ---
                                    # CSV logs every signal (pick helper sees the
                                    # full pool); email is reserved for the traits
                                    # that carried the edge in backtests.
                                    today_key = datetime.now().date()
                                    if email_day != today_key:
                                        email_day, emails_today, emails_by_sector = today_key, 0, {}

                                    ret_5d = details.get('stk_ret_5d', daily_ctx.get('ret_5d', 0.0))
                                    sec_key = str(sector).strip() or 'Unknown'
                                    skip_email = ""
                                    if quiet_period:
                                        skip_email = f"quiet period until picks email at {args.picks_time}"
                                    elif args.email_mild_dip_only and not (args.band_min < ret_5d < 0.0):
                                        skip_email = f"5d {ret_5d:+.1f}% not a mild dip ({args.band_min:g}%..0%)"
                                    elif emails_today >= args.max_emails_per_day:
                                        skip_email = f"daily email cap ({args.max_emails_per_day}) reached"
                                    elif emails_by_sector.get(sec_key, 0) >= args.max_emails_per_sector:
                                        skip_email = f"sector cap ({args.max_emails_per_sector}) reached for {sec_key}"

                                    if skip_email:
                                        LAST_ALERT_TS[tkr] = now_ts
                                        print(f"  📥 LOGGED [{score}pts]: {tkr} — no email: {skip_email}")
                                    else:
                                        regime_tag = f" [{regime.upper()}]" if regime == 'caution' else ""
                                        subj = f"BUY [{score}pts]{regime_tag}: {tkr} {company_name} — {args.interval}"
                                        body = _build_alert_body(
                                            tkr, company_name, sector, score,
                                            details, args.interval, effective_threshold,
                                        )

                                        try:
                                            send_email_alert(
                                                args.smtp_host, args.smtp_port,
                                                smtp_user=smtp_user, smtp_pass=smtp_pass,
                                                to_list=to_list, subject=subj, body=body,
                                            )
                                            LAST_ALERT_TS[tkr] = now_ts
                                            alerts_sent += 1
                                            emails_today += 1
                                            emails_by_sector[sec_key] = emails_by_sector.get(sec_key, 0) + 1
                                            print(f"  🚨 ALERT [{score}pts]: {tkr} at ${price:.2f} "
                                                  f"(email {emails_today}/{args.max_emails_per_day})")
                                        except Exception as ee:
                                            print(f"  ⚠️  Email send failed for {tkr}: {ee}")

                                    log_alert_to_csv(
                                        log_path, tkr, score, details,
                                        company_name, sector,
                                    )

                        LAST_SCORE[tkr] = score

                    except Exception as e:
                        print(f"  ❌ Error processing {tkr}: {str(e)[:60]}...")
                        continue

                # Console summary of top scorers
                scored_list.sort(key=lambda x: x[1], reverse=True)
                if scored_list:
                    print(f"\n  📊 Top signals (>= {effective_threshold}):")
                    for tkr, sc, det in scored_list[:10]:
                        flags = []
                        if det.get('above_sma200'):
                            flags.append('SMA200')
                        adx_t = det.get('adx_tier', '')
                        if adx_t in ('strong', 'trending'):
                            flags.append(f'ADX-{adx_t}')
                        if det.get('rs_outperforming'):
                            flags.append('RS')
                        rsi_z = det.get('rsi_zone', '')
                        if rsi_z == 'sweet_spot':
                            flags.append('RSI*')
                        elif rsi_z == 'warm':
                            flags.append('RSI')
                        if det.get('volume_surge'):
                            flags.append('VOL+')
                        if det.get('bb_lower_bounce'):
                            flags.append('BB-bounce')
                        mh = det.get('macd_hist', '')
                        if mh == 'strong':
                            flags.append('MACD+')
                        print(
                            f"    {tkr:6s}  score={sc:2d}/17"
                            f"  ${det.get('price', 0):8.2f}"
                            f"  RSI={det.get('rsi', 0):5.1f}"
                            f"  R:R={det.get('rr_ratio', 0):4.1f}"
                            f"  [{', '.join(flags)}]"
                        )

                print(
                    f"\n  ✅ Scan done [{regime.upper()}]. {alerts_sent} alerts sent,"
                    f" {len(scored_list)} above threshold ({effective_threshold})."
                    f" Next in {args.poll_secs}s"
                )

            else:
                print("⏸️  Market closed — sleeping…")

            time.sleep(max(15, int(args.poll_secs)))

    except KeyboardInterrupt:
        print("\n🛑 Live intraday alerts stopped by user.")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intraday multi-timeframe scored stock alerts.",
    )

    parser.add_argument("--cli", action="store_true",
                        help="Run in CLI mode (required).")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME_20,
                        help="Min 20-bar avg volume.")
    parser.add_argument("--min-price", type=float, default=MIN_LAST_CLOSE,
                        help="Min last close price.")
    parser.add_argument("--lookback", type=int, default=lookback_days,
                        help="Lookback bars for recent intraday signals.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help="Batch size for yfinance downloads.")
    parser.add_argument("--universe", type=str, default="sp500",
                        choices=["sp500", "nasdaq"],
                        help="Ticker universe to scan.")

    parser.add_argument("--interval", type=str, default="15m", required=True,
                        help="Intraday bar interval (1m, 5m, 15m, 30m, 1h)")
    parser.add_argument("--period", type=str, default="10d", required=True,
                        help="Intraday lookback period (5d, 10d, 30d)")
    parser.add_argument("--live", action="store_true", required=True,
                        help="Enable live polling loop (required).")
    parser.add_argument("--poll-secs", type=int, default=120,
                        help="Seconds between scans.")

    parser.add_argument("--smtp-host", type=str, default="smtp.gmail.com",
                        help="SMTP hostname.")
    parser.add_argument("--smtp-port", type=int, default=465,
                        help="SMTP port (465 = SSL).")
    parser.add_argument("--smtp-user", type=str, required=True,
                        help="SMTP username / email address.")
    parser.add_argument("--smtp-pass-env", type=str, default="SMTP_APP_PASSWORD",
                        help="Env var name holding SMTP password.")
    parser.add_argument("--email-to", type=str, required=True,
                        help="Comma-separated recipient emails.")

    parser.add_argument("--score-threshold", type=int, default=SCORE_THRESHOLD,
                        help="Minimum score to trigger alert (default: 11, max 17).")
    parser.add_argument("--debounce-mins", type=int, default=60,
                        help="Per-ticker alert cooldown in minutes (default: 60).")
    parser.add_argument("--alert-log", type=str, default="alert_log.csv",
                        help="CSV path for logging every alert (for backtesting).")

    parser.add_argument("--earnings-blackout", action=argparse.BooleanOptionalAction, default=True,
                        help="Veto buys held through earnings (default: on, for gap tail-risk).")
    parser.add_argument("--earnings-blackout-days", type=int, default=7,
                        help="Veto if earnings is within this many days (default: 7).")
    parser.add_argument("--news-veto", action=argparse.BooleanOptionalAction, default=True,
                        help="Veto buys with strongly negative recent news (default: on).")
    parser.add_argument("--news-lookback-hours", type=int, default=72,
                        help="How far back to scan headlines for sentiment (default: 72h).")
    parser.add_argument("--momentum-band", action=argparse.BooleanOptionalAction, default=True,
                        help="Veto buys whose 5-day move is outside the validated band (default: on).")
    parser.add_argument("--band-min", type=float, default=-6.0,
                        help="Lower bound of 5-day return band, %% (default: -6).")
    parser.add_argument("--band-max", type=float, default=6.0,
                        help="Upper bound of 5-day return band, %% (default: +6).")
    parser.add_argument("--sector-guard", action=argparse.BooleanOptionalAction, default=True,
                        help="Veto buys whose sector ETF is crashing (default: on).")
    parser.add_argument("--sector-min", type=float, default=-5.0,
                        help="Veto if sector ETF 5-day return <= this %% (default: -5).")
    parser.add_argument("--skip-caution", action=argparse.BooleanOptionalAction, default=True,
                        help="Suppress all alerts in caution regime (default: on; "
                             "backtest shows caution trades lose money).")
    parser.add_argument("--email-mild-dip-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Email only signals whose 5-day move is a mild dip "
                             "(band-min..0%%), where backtested edge concentrates. "
                             "Others are still logged to CSV (default: on).")
    parser.add_argument("--max-emails-per-day", type=int, default=6,
                        help="Hard cap on alert emails per day (default: 6).")
    parser.add_argument("--max-emails-per-sector", type=int, default=2,
                        help="Max alert emails per sector per day (default: 2).")
    parser.add_argument("--daily-picks", action=argparse.BooleanOptionalAction, default=True,
                        help="Hold alert emails until --picks-time, then send one ranked "
                             "DAILY PICKS email; live emails resume after (default: on).")
    parser.add_argument("--picks-time", type=str, default="07:30",
                        help="Local (Pacific) HH:MM to send the daily picks email (default: 07:30).")
    parser.add_argument("--picks-count", type=int, default=3,
                        help="Number of picks in the daily picks email (default: 3).")

    args = parser.parse_args()

    if not args.cli:
        print("This script requires --cli flag.")
        exit(1)

    run_cli(args)
