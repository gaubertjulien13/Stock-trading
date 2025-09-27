# file: stock_multi_method_screener.py

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from the same folder as the script:
load_dotenv(Path(__file__).with_name('.stock_screener.env'))

import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from io import StringIO
import io
import time
import json
import gzip
import smtplib
from email.message import EmailMessage
from textwrap import dedent

# Import universe ticker functions from myutils
from myutils import get_sp500_tickers, get_nasdaq_composite_tickers

# =========================
# Global config / cache
# =========================
PRICE_DATA = {}                 # Batch cache
BATCH_SIZE = 20                 # Tune for rate limits
MIN_AVG_VOLUME_20 = 150_000     # 0 to disable
MIN_LAST_CLOSE = 3.0            # 0 to disable

# Default time window
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
lookback_days = 5  # bars, not calendar days

# =========================
# Helper: chunking
# =========================
def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# Enhanced company info extraction
# =========================
def get_company_info_robust(ticker, max_retries=3):
    """
    Enhanced company info extraction with retries and better fallbacks.
    Returns (company_name, sector, market_cap) tuple.
    """
    import time

    for attempt in range(max_retries):
        try:
            ti = yf.Ticker(ticker)

            company_name = ticker
            sector = "N/A"
            market_cap = np.nan

            # Prefer newer yfinance .get_info(); fallback to .info
            info = {}
            try:
                if hasattr(ti, "get_info"):
                    info = ti.get_info() or {}
                else:
                    info = ti.info or {}
            except Exception:
                info = {}

            # Names
            name_candidate = info.get('longName') or info.get('shortName') or info.get('displayName')
            if name_candidate and str(name_candidate).strip():
                company_name = str(name_candidate).strip()

            # Sector / industry
            sec_candidate = info.get('sector') or info.get('industry')
            if sec_candidate and str(sec_candidate).strip():
                sector = str(sec_candidate).strip()

            # Market cap
            mc = info.get('marketCap')
            if mc is not None:
                try:
                    market_cap = float(mc)
                except Exception:
                    pass

            return company_name, sector, market_cap

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed for {ticker}: {str(e)[:50]}... Retrying...")
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                print(f"  ❌ All attempts failed for {ticker}. Using ticker as company name.")

    return ticker, "N/A", np.nan

# =========================
# Indicators
# =========================

def _extract_cols(df, ticker=None):
    """
    Return (close, high, low, volume) Series regardless of single or MultiIndex columns.
    After batch caching, per-ticker frames are single-level (fields only).
    """
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker is not None and ticker in lv0:
            sub = df[ticker]
        elif ticker is not None and ticker in lv1:
            sub = df.xs(ticker, axis=1, level=1)
        else:
            if set(['Close','Open','High','Low','Volume']).issubset(set(lv0)):
                sub = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
            else:
                sub = df.xs(df.columns.get_level_values(0)[0], axis=1, level=0)
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = sub.columns.get_level_values(0)
        close_col = sub['Close']
        high_col = sub['High']
        low_col  = sub['Low']
        volume_col = sub['Volume']
    else:
        close_col = df['Close']
        high_col  = df['High']
        low_col   = df['Low']
        volume_col = df['Volume']
    return close_col.astype(float), high_col.astype(float), low_col.astype(float), volume_col.astype(float)

# initial version with small fixes 
def calculate_indicators(ticker, start_date, end_date):
    """
    Calculate technical indicators for a given ticker.
    Uses preloaded PRICE_DATA if available (speeds up dramatically).
    """
    try:
        if ticker in PRICE_DATA:
            df = PRICE_DATA[ticker].copy()
        else:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by='ticker', threads=True)

        if df is None or df.empty or len(df) < 60:
            return None

        close_col, high_col, low_col, volume_col = _extract_cols(df, ticker=ticker)

        # === METHODOLOGY 1: SMA CROSSOVER ===
        df["SMA20"] = close_col.rolling(window=20, min_periods=20).mean()
        df["SMA50"] = close_col.rolling(window=50, min_periods=50).mean()
        df["SMA_Buy_Signal"] = (df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))

        # === (NEW) Trend Filter: 200-day SMA ===
        df["SMA200"] = close_col.rolling(window=200, min_periods=200).mean()

        # === METHODOLOGY 2: EMA & RSI STRATEGY ===
        df["EMA5"] = close_col.ewm(span=5, adjust=False).mean()
        df["EMA20"] = close_col.ewm(span=20, adjust=False).mean()

        delta = close_col.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"] = df["RSI"].fillna(50)

        # EMA & RSI Buy: EMA5 > EMA20 AND 50 < RSI < 70
        df["EMA_RSI_Buy_Signal"] = (df["EMA5"] > df["EMA20"]) & (df["RSI"] > 50) & (df["RSI"] < 70)

        # === METHODOLOGY 3: BOLLINGER BANDS ===
        df['BB_Middle'] = close_col.rolling(20, min_periods=20).mean()
        bb_std = close_col.rolling(20, min_periods=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']).abs()

        bb_lower_bounce = (close_col.shift(1) <= df['BB_Lower'].shift(1) * 1.02) & (close_col > df['BB_Lower'] * 1.02)
        bb_middle_cross = (close_col > df['BB_Middle']) & (close_col.shift(1) <= df['BB_Middle'].shift(1))
        bb_width_ok = df['BB_Width'] > df['BB_Width'].rolling(20, min_periods=20).mean() * 0.8

        df["BB_Buy_Signal"] = (bb_lower_bounce | bb_middle_cross) & bb_width_ok

        # === EXTRAS ===
        prev_close = close_col.shift(1)
        tr1 = high_col - low_col
        tr2 = (high_col - prev_close).abs()
        tr3 = (low_col - prev_close).abs()
        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR14"] = df["TR"].rolling(14, min_periods=14).mean()

        df['Volume_MA'] = volume_col.rolling(20, min_periods=20).mean()
        df['Volume_Above_Avg'] = volume_col > df['Volume_MA']

        df["BB_Range"] = (df['BB_Upper'] - df['BB_Lower']).abs().clip(lower=1e-9)

        # === (NEW) MACD (12-26-9) ===
        ema12 = close_col.ewm(span=12, adjust=False).mean()
        ema26 = close_col.ewm(span=26, adjust=False).mean()
        df["MACD_Line"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

        df = df.dropna()
        if df.empty:
            return None

        # === COMBINED BUY SIGNAL ===
        df["All_Methods_Buy"] = (
            df["SMA_Buy_Signal"] | (df["SMA20"] > df["SMA50"])
        ) & df["EMA_RSI_Buy_Signal"] & df["BB_Buy_Signal"]

        df["Quality_Buy_Signal"] = df["All_Methods_Buy"] & df["Volume_Above_Avg"]

        return df

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

# =========================
# Batch slice helper (FIXED)
# =========================
def _slice_from_batch(data, tkr):
    """
    Robustly extract a single-ticker OHLCV frame from a yfinance batch DataFrame,
    regardless of whether columns are (field, ticker) or (ticker, field).
    Returns a single-level dataframe with columns ['Open','High','Low','Close','Adj Close','Volume'] when available.
    """
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

    cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in sub.columns]
    if not cols:
        return None
    sub = sub[cols].dropna(how='all')
    return sub if not sub.empty else None

# =========================
# Intraday strict-now helpers (flip + alert)
# =========================
def _strict_now_flags(df, lookback_bars=5):
    """
    Returns (base_now, strict_gate_now, last_base_buy_idx)
    base_now = All_Methods_Buy on the current bar (ungated)
    strict_gate_now = SMA200 gate now AND (MACD bull now OR recent cross up in last N bars)
    """
    if df is None or df.empty:
        return False, False, None

    N = max(3, int(lookback_bars))
    recent = df.tail(N)

    # Base (ungated) now
    base_now = bool(df["All_Methods_Buy"].fillna(False).iloc[-1])

    # Trend & MACD gates (now or recent cross up)
    sma200_now = df["SMA200"].iloc[-1] if "SMA200" in df.columns else np.nan
    close_now  = df["Close"].iloc[-1]
    trend_ok_now = (pd.notna(sma200_now) and (close_now > sma200_now))

    macd_now_ok = False
    if {'MACD_Line','MACD_Signal'}.issubset(df.columns):
        macd_now_ok = pd.notna(df["MACD_Line"].iloc[-1]) and pd.notna(df["MACD_Signal"].iloc[-1]) and \
                      (df["MACD_Line"].iloc[-1] > df["MACD_Signal"].iloc[-1])

    macd_cross_up_recent = False
    if {'MACD_Line','MACD_Signal'}.issubset(recent.columns):
        macd_cross_up_recent = (
            (recent["MACD_Line"] > recent["MACD_Signal"]) &
            (recent["MACD_Line"].shift(1) <= recent["MACD_Signal"].shift(1))
        ).any()

    strict_gate_now = bool(trend_ok_now and (macd_now_ok or macd_cross_up_recent))

    # Last base buy within lookback
    last_base_buy_idx = recent.index[recent["All_Methods_Buy"].fillna(False)].max() \
                        if recent["All_Methods_Buy"].fillna(False).any() else None

    return base_now, strict_gate_now, last_base_buy_idx

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

def _market_open_pacific_now():
    # Simple check for Mon–Fri, 06:30–13:00 PT; for robust DST handling, use zoneinfo/pytz
    from datetime import datetime, time as dt_time, timedelta, timezone
    PT = timezone(timedelta(hours=-7))  # assumes PDT; adjust if needed
    now = datetime.now(PT)
    if now.weekday() >= 5:
        return False
    open_t = dt_time(6, 30)
    close_t = dt_time(13, 0)
    return (now.time() >= open_t) and (now.time() <= close_t)

# =========================
# CLI entrypoint (now includes live intraday alert loop)
# =========================
def run_cli(args):
    global MIN_AVG_VOLUME_20, MIN_LAST_CLOSE, lookback_days, BATCH_SIZE
    MIN_AVG_VOLUME_20 = args.min_volume
    MIN_LAST_CLOSE = args.min_price
    lookback_days = args.lookback
    BATCH_SIZE = args.batch

    # Live intraday mode only
    if not args.live:
        print("❌ This script is for intraday alerts only. Use --live flag.")
        return

    # Email config (prefer CLI flags; fall back to .env)
    smtp_user = args.smtp_user or os.environ.get("ALERT_FROM_EMAIL", "")
    to_list_raw = args.email_to or os.environ.get("ALERT_TO_EMAILS", "")
    to_list = [e.strip() for e in to_list_raw.split(",") if e.strip()]

    if not to_list:
        print("⚠️  --live requires recipients. Provide --email-to or set ALERT_TO_EMAILS in .stock_screener.env")
        return

    smtp_pass = os.environ.get(args.smtp_pass_env, "")

    print(f"🔄 Live intraday alerts: interval={args.interval}, period={args.period}, poll every {args.poll_secs}s")
    print(f"📧 Email to: {to_list} via {args.smtp_host}:{args.smtp_port} as {smtp_user or '(no user)'}")

    LAST_STRICT = {}     # {ticker: bool} strict condition last seen
    LAST_ALERT_TS = {}   # debounce per ticker (epoch seconds)

    # Universe selection once
    if args.universe.lower() == "nasdaq":
        tickers = get_nasdaq_composite_tickers()
    else:
        tickers = get_sp500_tickers()

    print(f"📊 Monitoring {len(tickers)} tickers for strict buy signals...")

    try:
        while True:
            if _market_open_pacific_now():
                print(f"🔍 Scanning at {datetime.now().strftime('%H:%M:%S')}...")
                
                # Download intraday data in batches
                PRICE_DATA.clear()
                for i, batch in enumerate(_chunks(tickers, BATCH_SIZE), start=1):
                    try:
                        print(f"  📥 Downloading batch {i}: {len(batch)} tickers...")
                        
                        data = yf.download(
                            batch,
                            period=args.period,
                            interval=args.interval,
                            progress=False,
                            group_by='ticker',
                            threads=True
                        )
                        
                        if data is not None and not data.empty:
                            if isinstance(data.columns, pd.MultiIndex):
                                for tkr in batch:
                                    try:
                                        sub = _slice_from_batch(data, tkr)
                                        if sub is not None and not sub.empty:
                                            PRICE_DATA[tkr] = sub
                                    except Exception as e:
                                        print(f"    ⚠️  Failed to extract {tkr}: {e}")
                            else:
                                # Single ticker case
                                if len(batch) == 1:
                                    PRICE_DATA[batch[0]] = data.dropna().copy()
                                    
                    except Exception as e:
                        print(f"    ❌ Batch download error: {e}")
                        continue
                
                now_ts = time.time()
                alerts_sent = 0
                
                # Check each ticker for flip conditions
                for tkr in tickers:
                    try:
                        if tkr not in PRICE_DATA:
                            continue
                            
                        # Apply liquidity filters
                        df_raw = PRICE_DATA[tkr]
                        if len(df_raw) < 20:
                            continue
                            
                        vol_ok = (MIN_AVG_VOLUME_20 <= 0 or 
                                 float(df_raw['Volume'].tail(20).mean()) >= float(MIN_AVG_VOLUME_20))
                        px_ok = (MIN_LAST_CLOSE <= 0 or 
                                float(df_raw['Close'].iloc[-1]) >= float(MIN_LAST_CLOSE))
                        
                        if not (vol_ok and px_ok):
                            continue
                        
                        # Calculate indicators and check for alerts
                        df2 = calculate_indicators(tkr, start_date, end_date)
                        if df2 is None or df2.empty:
                            continue

                        base_now, strict_gate_now, last_base_buy_idx = _strict_now_flags(df2, lookback_bars=lookback_days)
                        strict_now = bool(base_now and strict_gate_now)

                        prev = LAST_STRICT.get(tkr, False)
                        flipped = (not prev) and strict_now

                        # Debounce: 30 minutes per ticker
                        if flipped:
                            last_sent = LAST_ALERT_TS.get(tkr, 0)
                            if now_ts - last_sent >= 30 * 60:
                                price = float(df2["Close"].iloc[-1])
                                rsi   = float(df2["RSI"].iloc[-1]) if "RSI" in df2.columns and pd.notna(df2["RSI"].iloc[-1]) else float("nan")
                                ema_s = float(((df2["EMA5"].iloc[-1] - df2["EMA20"].iloc[-1]) / df2["EMA20"].iloc[-1]) * 100.0) if "EMA5" in df2.columns and "EMA20" in df2.columns and df2["EMA20"].iloc[-1] else float("nan")
                                sma_s = float(((df2["SMA20"].iloc[-1] - df2["SMA50"].iloc[-1]) / df2["SMA50"].iloc[-1]) * 100.0) if "SMA20" in df2.columns and "SMA50" in df2.columns and df2["SMA50"].iloc[-1] else float("nan")

                                # Company info (best-effort)
                                try:
                                    company_name, sector, _ = get_company_info_robust(tkr, max_retries=1)
                                except Exception:
                                    company_name, sector = tkr, "N/A"

                                subj = f"🚨 STRICT BUY ALERT: {tkr} {company_name} — {args.interval}"
                                body = dedent(f"""\
                                    Strict multi-method BUY signal detected!

                                    Ticker:       {tkr}
                                    Company:      {company_name}
                                    Sector:       {sector}
                                    Interval:     {args.interval}
                                    Price:        ${price:.2f}
                                    RSI:          {rsi:.1f}
                                    EMA strength: {ema_s:.2f}%
                                    SMA strength: {sma_s:.2f}%
                                    Signal time:  {last_base_buy_idx}

                                    Conditions met:
                                      ✅ All methodologies fired (SMA + EMA/RSI + Bollinger)
                                      ✅ Above SMA200 trend filter
                                      ✅ MACD bullish confirmation
                                      
                                    This alert fires when all conditions flip from FALSE to TRUE.
                                """)

                                try:
                                    send_email_alert(
                                        args.smtp_host, args.smtp_port,
                                        smtp_user=smtp_user, smtp_pass=smtp_pass,
                                        to_list=to_list, subject=subj, body=body
                                    )
                                    LAST_ALERT_TS[tkr] = now_ts
                                    alerts_sent += 1
                                    print(f"🚨 ALERT SENT: {tkr} at ${price:.2f}")
                                except Exception as ee:
                                    print(f"⚠️  Email send failed for {tkr}: {ee}")

                        LAST_STRICT[tkr] = strict_now
                        
                    except Exception as e:
                        print(f"❌ Error processing {tkr}: {str(e)[:50]}...")
                        continue
                
                print(f"✅ Scan complete. {alerts_sent} alerts sent. Next scan in {args.poll_secs}s")
                
            else:
                print("⏸️  Market closed (PT) — sleeping…")

            time.sleep(max(15, int(args.poll_secs)))
            
    except KeyboardInterrupt:
        print("\n🛑 Live intraday alerts stopped by user.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday multi-method stock alerts.")
    
    # Basic CLI arguments
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (required for this script).")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME_20, help="Min 20-day avg volume.")
    parser.add_argument("--min-price", type=float, default=MIN_LAST_CLOSE, help="Min last close price.")
    parser.add_argument("--lookback", type=int, default=lookback_days, help="Lookback bars for signals.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size for yfinance download.")
    parser.add_argument("--universe", type=str, default="sp500", choices=["sp500", "nasdaq"],
                        help="Universe to scan (sp500 or nasdaq).")

    # Intraday + Live flags (required)
    parser.add_argument("--interval", type=str, default="15m", required=True,
                        help="yfinance interval for intraday alerts (e.g. 1m, 5m, 15m, 30m, 1h)")
    parser.add_argument("--period", type=str, default="10d", required=True,
                        help="yfinance period for intraday alerts (e.g. 5d, 10d, 30d)")
    parser.add_argument("--live", action="store_true", required=True,
                        help="Enable live polling loop for intraday alerts (required)")
    parser.add_argument("--poll-secs", type=int, default=120,
                        help="Seconds between scans when --live is on")

    # Email alert config (required for alerts)
    parser.add_argument("--smtp-host", type=str, default="smtp.gmail.com",
                        help="SMTP hostname (default: Gmail SMTP)")
    parser.add_argument("--smtp-port", type=int, default=465,
                        help="SMTP port (465 SSL recommended)")
    parser.add_argument("--smtp-user", type=str, required=True,
                        help="SMTP username / email address")
    parser.add_argument("--smtp-pass-env", type=str, default="SMTP_APP_PASSWORD",
                        help="Name of env var holding your SMTP password / app password")
    parser.add_argument("--email-to", type=str, required=True,
                        help="Comma-separated recipient emails for alerts")

    args = parser.parse_args()

    # Force CLI mode for this script
    if not args.cli:
        print("❌ This script requires --cli flag. Use: python script.py --cli --live ...")
        exit(1)

    # Run the CLI function
    run_cli(args)