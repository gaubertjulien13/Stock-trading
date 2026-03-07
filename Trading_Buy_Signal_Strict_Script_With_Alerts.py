# file: Trading_Buy_Signal_Strict_Script_With_Alerts.py
#
# Multi-timeframe scored intraday alert system.
# Daily bars provide trend context (SMA200, MACD, ADX, relative strength vs SPY).
# Intraday bars provide entry timing (EMA crossover, RSI, Bollinger, volume surge).
# A point-based scoring system replaces binary signals for better ranking.

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

BATCH_SIZE = 20
MIN_AVG_VOLUME_20 = 150_000
MIN_LAST_CLOSE = 3.0
SCORE_THRESHOLD = 8

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


def refresh_daily_context(tickers, batch_size=20):
    """
    Download daily bars and compute per-ticker trend context:
    SMA200, daily MACD, ADX, ATR, and 20-day relative strength vs SPY.
    Cached for the calendar day — skipped if already fresh.
    """
    global DAILY_DATA, DAILY_CONTEXT, DAILY_CACHE_DATE, SPY_DAILY_CLOSE

    today = datetime.today().date()
    if DAILY_CACHE_DATE == today and DAILY_CONTEXT:
        return

    print("📅 Refreshing daily trend context (runs once per trading day)...")
    DAILY_DATA.clear()
    DAILY_CONTEXT.clear()

    daily_start = datetime.today() - timedelta(days=400)
    daily_end = datetime.today()

    # SPY for relative-strength benchmark
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

            # -- Daily MACD (12-26-9) --
            ema12 = close_col.ewm(span=12, adjust=False).mean()
            ema26 = close_col.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal

            daily_macd_bullish = bool(
                pd.notna(macd_line.iloc[-1]) and
                pd.notna(macd_signal.iloc[-1]) and
                macd_line.iloc[-1] > macd_signal.iloc[-1]
            )
            daily_macd_hist_rising = bool(
                len(macd_hist.dropna()) >= 2 and
                macd_hist.iloc[-1] > macd_hist.iloc[-2]
            )

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

            DAILY_CONTEXT[tkr] = {
                'above_sma200': above_sma200,
                'sma200': sma200_val,
                'daily_close': latest_close,
                'daily_macd_bullish': daily_macd_bullish,
                'daily_macd_hist_rising': daily_macd_hist_rising,
                'adx': adx_val,
                'adx_strong': adx_val > 25,
                'daily_atr': daily_atr,
                'rs_outperforming': rs_outperforming,
            }
        except Exception:
            continue

    DAILY_CACHE_DATE = today
    print(f"  ✅ Daily context ready for {len(DAILY_CONTEXT)} tickers")


# =========================
# Intraday indicators (computed on intraday bars for entry timing)
# =========================

def calculate_intraday_indicators(ticker):
    """
    Compute intraday entry-timing indicators: EMA crossover, Wilder RSI,
    Bollinger Bands, volume surge, intraday MACD with histogram direction.
    """
    try:
        if ticker not in PRICE_DATA:
            return None

        df = PRICE_DATA[ticker].copy()
        if df is None or df.empty or len(df) < 40:
            return None

        close, high, low, volume = _extract_cols(df, ticker=ticker)

        # -- EMA crossover --
        df["EMA5"] = close.ewm(span=5, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()

        # -- SMA crossover (intraday) with slope check --
        df["SMA20"] = close.rolling(20, min_periods=20).mean()
        if len(df) >= 50:
            df["SMA50"] = close.rolling(50, min_periods=50).mean()
            df["SMA_Cross_Up"] = (
                (df["SMA20"] > df["SMA50"]) &
                (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
            )
            df["SMA20_rising"] = df["SMA20"] > df["SMA20"].shift(3)
            df["SMA_Trend_Up"] = df["SMA20_rising"] & (df["SMA20"] > df["SMA50"])
        else:
            df["SMA50"] = np.nan
            df["SMA_Cross_Up"] = False
            df["SMA20_rising"] = df["SMA20"] > df["SMA20"].shift(3)
            df["SMA_Trend_Up"] = df["SMA20_rising"]

        # -- RSI (Wilder's exponential smoothing) --
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        df["EMA_RSI_Buy"] = (
            (df["EMA5"] > df["EMA20"]) &
            df["RSI"].between(50, 70, inclusive="neither")
        )

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

        # -- MACD (intraday 12-26-9) with histogram direction --
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD_Line"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
        df["MACD_Bullish"] = df["MACD_Line"] > df["MACD_Signal"]
        df["MACD_Hist_Rising"] = df["MACD_Hist"] > df["MACD_Hist"].shift(1)

        # -- BB position (0% = lower band, 100% = upper band) --
        df["BB_Position"] = ((close - df["BB_Lower"]) / df["BB_Range"]) * 100.0

        df = df.dropna(subset=["RSI", "EMA5", "EMA20", "BB_Middle"])
        return df if not df.empty else None

    except Exception as e:
        print(f"  Error computing intraday indicators for {ticker}: {e}")
        return None


# =========================
# Scoring system
# =========================
#
# Maximum theoretical score breakdown:
#   Daily:    above_sma200(2) + macd_bull(2) + macd_hist_rising(1)
#             + adx_strong(2) + rs_spy(1) = 8
#   Intraday: sma_cross(2)|sma_trend(1) + ema_rsi(2|1)
#             + bb_bounce(3)|bb_cross(1) + vol_surge(2)|vol_above(1)
#             + macd_hist_rising(1) + macd_bull(1) = 11
#   Grand max ≈ 19

def compute_signal_score(intra_df, daily_ctx, lookback_bars=5):
    """
    Combine daily trend context with intraday entry signals into a numeric
    score.  Returns (score, details_dict) or (0, None).
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

    # ===== Daily context points =====

    if daily_ctx.get('above_sma200', False):
        score += 2
        details['above_sma200'] = True
    else:
        details['above_sma200'] = False

    if daily_ctx.get('daily_macd_bullish', False):
        score += 2
        details['daily_macd_bullish'] = True
        if daily_ctx.get('daily_macd_hist_rising', False):
            score += 1
            details['daily_macd_hist_rising'] = True
    else:
        details['daily_macd_bullish'] = False

    if daily_ctx.get('adx_strong', False):
        score += 2
        details['adx_strong'] = True
    else:
        details['adx_strong'] = False
    details['adx'] = daily_ctx.get('adx', 0.0)

    if daily_ctx.get('rs_outperforming', False):
        score += 1
        details['rs_outperforming'] = True
    else:
        details['rs_outperforming'] = False

    # ===== Intraday signal points =====

    # SMA crossover in recent N bars (+2) or rising SMA trend (+1)
    if "SMA_Cross_Up" in recent.columns and recent["SMA_Cross_Up"].any():
        score += 2
        details['sma_cross_recent'] = True
    elif pd.notna(now.get("SMA_Trend_Up")) and bool(now["SMA_Trend_Up"]):
        score += 1
        details['sma_trend_up'] = True

    # EMA/RSI: RSI 50-65 (+2), RSI 65-70 (+1)
    rsi_val = float(now["RSI"]) if pd.notna(now.get("RSI")) else np.nan
    details['rsi'] = rsi_val
    if pd.notna(now.get("EMA_RSI_Buy")) and bool(now["EMA_RSI_Buy"]):
        if pd.notna(rsi_val) and rsi_val < 65:
            score += 2
            details['ema_rsi_buy'] = 'strong'
        else:
            score += 1
            details['ema_rsi_buy'] = 'moderate'

    # Bollinger: lower bounce (+3) or middle cross (+1)
    if "BB_Lower_Bounce" in recent.columns and recent["BB_Lower_Bounce"].any():
        score += 3
        details['bb_lower_bounce'] = True
    elif "BB_Buy" in recent.columns and recent["BB_Buy"].any():
        score += 1
        details['bb_middle_cross'] = True

    # Volume surge >1.5x (+2) or above average (+1)
    if pd.notna(now.get("Volume_Surge")) and bool(now["Volume_Surge"]):
        score += 2
        details['volume_surge'] = True
    elif pd.notna(now.get("Volume_Above_Avg")) and bool(now["Volume_Above_Avg"]):
        score += 1
        details['volume_above_avg'] = True

    # Intraday MACD histogram rising (+1)
    if pd.notna(now.get("MACD_Hist_Rising")) and bool(now["MACD_Hist_Rising"]):
        score += 1
        details['intraday_macd_hist_rising'] = True

    # Intraday MACD bullish (+1)
    if pd.notna(now.get("MACD_Bullish")) and bool(now["MACD_Bullish"]):
        score += 1
        details['intraday_macd_bullish'] = True

    # ===== Risk / reward context (daily ATR for proper position sizing) =====

    price = float(now["Close"]) if pd.notna(now.get("Close")) else np.nan
    daily_atr = daily_ctx.get('daily_atr', np.nan)

    if pd.notna(price) and pd.notna(daily_atr) and daily_atr > 0:
        stop_1_5x = price - 1.5 * daily_atr
        stop_2x = price - 2.0 * daily_atr
        target_2x = price + 2.0 * daily_atr
        target_3x = price + 3.0 * daily_atr
        atr_pct = (daily_atr / price) * 100.0
        risk = price - stop_1_5x
        rr_ratio = (target_3x - price) / risk if risk > 0 else 0.0
    else:
        stop_1_5x = stop_2x = target_2x = target_3x = np.nan
        atr_pct = rr_ratio = np.nan

    ema5 = float(now["EMA5"]) if pd.notna(now.get("EMA5")) else np.nan
    ema20 = float(now["EMA20"]) if pd.notna(now.get("EMA20")) else np.nan
    sma20 = float(now["SMA20"]) if pd.notna(now.get("SMA20")) else np.nan
    bb_pos = float(now["BB_Position"]) if pd.notna(now.get("BB_Position")) else np.nan

    details.update({
        'price': price,
        'ema5': ema5,
        'ema20': ema20,
        'sma20': sma20,
        'bb_position': bb_pos,
        'daily_atr': daily_atr,
        'atr_pct': atr_pct,
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
    'timestamp', 'ticker', 'company', 'sector', 'score',
    'price', 'rsi', 'daily_atr', 'atr_pct',
    'stop_1_5x', 'stop_2x', 'target_2x', 'target_3x', 'rr_ratio',
    'above_sma200', 'daily_macd_bullish', 'adx', 'adx_strong',
    'rs_outperforming', 'volume_surge', 'bb_lower_bounce',
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

        writer.writerow([
            datetime.now().isoformat(),
            ticker,
            company_name,
            sector,
            score,
            _fmt(details.get('price'), 2),
            _fmt(details.get('rsi'), 1),
            _fmt(details.get('daily_atr'), 2),
            _fmt(details.get('atr_pct'), 2),
            _fmt(details.get('stop_1_5x'), 2),
            _fmt(details.get('stop_2x'), 2),
            _fmt(details.get('target_2x'), 2),
            _fmt(details.get('target_3x'), 2),
            _fmt(details.get('rr_ratio'), 1),
            details.get('above_sma200', ''),
            details.get('daily_macd_bullish', ''),
            _fmt(details.get('adx'), 1),
            details.get('adx_strong', ''),
            details.get('rs_outperforming', ''),
            details.get('volume_surge', ''),
            details.get('bb_lower_bounce', ''),
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


# =========================
# Build alert email body
# =========================

def _build_alert_body(tkr, company_name, sector, score, details, interval, threshold):
    price = details.get('price', 0)
    ema5 = details.get('ema5', 0)
    ema20 = details.get('ema20', 1)
    ema_spread = ((ema5 - ema20) / max(abs(ema20), 0.01)) * 100.0

    checks = []
    if details.get('above_sma200'):
        checks.append("  [+2] Above daily SMA200")
    else:
        checks.append("  [--] Below daily SMA200")

    if details.get('daily_macd_bullish'):
        checks.append("  [+2] Daily MACD bullish")
        if details.get('daily_macd_hist_rising'):
            checks.append("  [+1] Daily MACD histogram accelerating")
    else:
        checks.append("  [--] Daily MACD bearish")

    if details.get('adx_strong'):
        checks.append(f"  [+2] ADX strong ({details.get('adx', 0):.1f})")
    else:
        checks.append(f"  [--] ADX weak ({details.get('adx', 0):.1f})")

    if details.get('rs_outperforming'):
        checks.append("  [+1] Outperforming SPY (20-day)")
    else:
        checks.append("  [--] Underperforming SPY (20-day)")

    if details.get('sma_cross_recent'):
        checks.append("  [+2] SMA 20/50 crossover (recent)")
    elif details.get('sma_trend_up'):
        checks.append("  [+1] SMA20 rising above SMA50")

    ema_label = details.get('ema_rsi_buy', '')
    if ema_label == 'strong':
        checks.append(f"  [+2] EMA5>EMA20 + RSI {details.get('rsi', 0):.1f} (sweet spot)")
    elif ema_label == 'moderate':
        checks.append(f"  [+1] EMA5>EMA20 + RSI {details.get('rsi', 0):.1f} (getting warm)")

    if details.get('bb_lower_bounce'):
        checks.append("  [+3] Bollinger lower-band bounce")
    elif details.get('bb_middle_cross'):
        checks.append("  [+1] Bollinger middle-band cross")

    if details.get('volume_surge'):
        checks.append("  [+2] Volume surge (>1.5x 20-day avg)")
    elif details.get('volume_above_avg'):
        checks.append("  [+1] Volume above 20-day average")
    else:
        checks.append("  [--] Volume below average")

    if details.get('intraday_macd_hist_rising'):
        checks.append("  [+1] Intraday MACD momentum rising")
    if details.get('intraday_macd_bullish'):
        checks.append("  [+1] Intraday MACD bullish")

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
        Score:        {score} / ~19  (threshold: {threshold})
        Interval:     {interval}

        === Price & Momentum ===
        Price:            ${_safe(price)}
        RSI (Wilder 14):  {_safe(details.get('rsi'), '.1f')}
        EMA 5/20 spread:  {ema_spread:.2f}%
        BB position:      {_safe(details.get('bb_position'), '.1f')}%

        === Risk Management (based on daily ATR) ===
        Daily ATR:     ${_safe(details.get('daily_atr'))}  ({_safe(details.get('atr_pct'), '.1f')}% of price)
        Stop  1.5x:    ${_safe(details.get('stop_1_5x'))}
        Stop  2.0x:    ${_safe(details.get('stop_2x'))}
        Target 2.0x:   ${_safe(details.get('target_2x'))}
        Target 3.0x:   ${_safe(details.get('target_3x'))}
        R:R (3x/1.5x): {_safe(details.get('rr_ratio'), '.1f')} : 1

        === Signal Checklist ===
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
    print(f"🎯 Score threshold: {SCORE_THRESHOLD} / ~19, debounce: {args.debounce_mins}min")

    LAST_SCORE = {}
    LAST_ALERT_TS = {}
    log_path = args.alert_log

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
                        crossed_threshold = (prev_score < SCORE_THRESHOLD) and (score >= SCORE_THRESHOLD)

                        if score >= SCORE_THRESHOLD:
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

                                subj = f"BUY [{score}pts]: {tkr} {company_name} — {args.interval}"
                                body = _build_alert_body(
                                    tkr, company_name, sector, score,
                                    details, args.interval, SCORE_THRESHOLD,
                                )

                                try:
                                    send_email_alert(
                                        args.smtp_host, args.smtp_port,
                                        smtp_user=smtp_user, smtp_pass=smtp_pass,
                                        to_list=to_list, subject=subj, body=body,
                                    )
                                    LAST_ALERT_TS[tkr] = now_ts
                                    alerts_sent += 1
                                    print(f"  🚨 ALERT [{score}pts]: {tkr} at ${price:.2f}")
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
                    print(f"\n  📊 Top signals (>= {SCORE_THRESHOLD}):")
                    for tkr, sc, det in scored_list[:10]:
                        flags = []
                        if det.get('above_sma200'):
                            flags.append('SMA200')
                        if det.get('daily_macd_bullish'):
                            flags.append('MACD')
                        if det.get('adx_strong'):
                            flags.append('ADX')
                        if det.get('rs_outperforming'):
                            flags.append('RS')
                        if det.get('volume_surge'):
                            flags.append('VOL+')
                        if det.get('bb_lower_bounce'):
                            flags.append('BB-bounce')
                        print(
                            f"    {tkr:6s}  score={sc:2d}"
                            f"  ${det.get('price', 0):8.2f}"
                            f"  RSI={det.get('rsi', 0):5.1f}"
                            f"  R:R={det.get('rr_ratio', 0):4.1f}"
                            f"  [{', '.join(flags)}]"
                        )

                print(
                    f"\n  ✅ Scan done. {alerts_sent} alerts sent,"
                    f" {len(scored_list)} above threshold."
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
                        help="Minimum score to trigger alert (default: 8, max ~19).")
    parser.add_argument("--debounce-mins", type=int, default=60,
                        help="Per-ticker alert cooldown in minutes (default: 60).")
    parser.add_argument("--alert-log", type=str, default="alert_log.csv",
                        help="CSV path for logging every alert (for backtesting).")

    args = parser.parse_args()

    if not args.cli:
        print("This script requires --cli flag.")
        exit(1)

    run_cli(args)
