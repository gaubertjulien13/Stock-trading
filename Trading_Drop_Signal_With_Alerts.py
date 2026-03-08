# file: Trading_Sudden_Drop_Alert_Script.py

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
from datetime import datetime, timedelta, time as dt_time, timezone
import time
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

# Default thresholds
DROP_THRESHOLD_PCT = 10.0        # Percentage drop to trigger alert (e.g., 5%)
LOOKBACK_BARS = 5               # Lookback period for calculating drop
VOLUME_SPIKE_MULTIPLIER = 1.5   # Volume must be X times average to trigger

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

            info = {}
            try:
                if hasattr(ti, "get_info"):
                    info = ti.get_info() or {}
                else:
                    info = ti.info or {}
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
                print(f"  ⚠️  Attempt {attempt + 1} failed for {ticker}: {str(e)[:50]}... Retrying...")
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                print(f"  ❌ All attempts failed for {ticker}. Using ticker as company name.")

    return ticker, "N/A", np.nan

# =========================
# Data extraction helper
# =========================
def _extract_cols(df, ticker=None):
    """
    Return (close, high, low, volume) Series regardless of single or MultiIndex columns.
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

# =========================
# Drop detection function
# =========================
def detect_sudden_drop(ticker, drop_threshold_pct, lookback_bars, volume_spike_mult, require_volume_spike=False):
    """
    Detect sudden price drops for a given ticker.
    Returns dict with drop info or None if no drop detected.
    """
    try:
        if ticker not in PRICE_DATA:
            return None
            
        df = PRICE_DATA[ticker].copy()
        
        if df is None or df.empty or len(df) < lookback_bars + 10:
            return None

        close_col, high_col, low_col, volume_col = _extract_cols(df, ticker=ticker)
        
        # Calculate percentage change over lookback period
        current_price = float(close_col.iloc[-1])
        lookback_price = float(close_col.iloc[-lookback_bars-1]) if len(close_col) > lookback_bars else float(close_col.iloc[0])
        price_change_pct = ((current_price - lookback_price) / lookback_price) * 100.0
        
        # Also check single-bar drop
        prev_price = float(close_col.iloc[-2]) if len(close_col) > 1 else current_price
        single_bar_pct = ((current_price - prev_price) / prev_price) * 100.0
        
        # Calculate volume metrics
        volume_ma = volume_col.rolling(window=20, min_periods=10).mean()
        current_volume = float(volume_col.iloc[-1])
        avg_volume = float(volume_ma.iloc[-1]) if pd.notna(volume_ma.iloc[-1]) else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check if drop threshold is met
        drop_detected = False
        drop_type = None
        drop_pct = 0.0
        
        # Single bar drop
        if single_bar_pct <= -drop_threshold_pct:
            drop_detected = True
            drop_type = "single_bar"
            drop_pct = single_bar_pct
        # Multi-bar drop
        elif price_change_pct <= -drop_threshold_pct:
            drop_detected = True
            drop_type = f"{lookback_bars}_bar"
            drop_pct = price_change_pct
        
        if not drop_detected:
            return None
        
        # Volume spike requirement (optional)
        if require_volume_spike and volume_ratio < volume_spike_mult:
            return None
        
        # Calculate additional metrics
        high_52w = float(high_col.rolling(window=min(252, len(high_col))).max().iloc[-1]) if len(high_col) > 20 else current_price
        low_52w = float(low_col.rolling(window=min(252, len(low_col))).min().iloc[-1]) if len(low_col) > 20 else current_price
        price_from_52w_high = ((current_price - high_52w) / high_52w) * 100.0
        price_from_52w_low = ((current_price - low_52w) / low_52w) * 100.0
        
        # Calculate RSI for context
        delta = close_col.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'prev_price': prev_price,
            'drop_pct': drop_pct,
            'drop_type': drop_type,
            'volume_ratio': volume_ratio,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'price_from_52w_high': price_from_52w_high,
            'price_from_52w_low': price_from_52w_low,
            'rsi': current_rsi,
            'high_52w': high_52w,
            'low_52w': low_52w
        }
        
    except Exception as e:
        print(f"Error detecting drop for {ticker}: {e}")
        return None

# =========================
# Batch slice helper
# =========================
def _slice_from_batch(data, tkr):
    """
    Robustly extract a single-ticker OHLCV frame from a yfinance batch DataFrame.
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
# Email alert function
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
# Market hours check
# =========================
def _market_open_pacific_now():
    """Check if market is open (Mon-Fri, 06:30-13:00 PT)"""
    PT = timezone(timedelta(hours=-7))  # assumes PDT; adjust if needed
    now = datetime.now(PT)
    if now.weekday() >= 5:
        return False
    open_t = dt_time(6, 30)
    close_t = dt_time(13, 0)
    return (now.time() >= open_t) and (now.time() <= close_t)

# =========================
# CLI entrypoint
# =========================
def run_cli(args):
    global MIN_AVG_VOLUME_20, MIN_LAST_CLOSE, BATCH_SIZE, DROP_THRESHOLD_PCT, LOOKBACK_BARS, VOLUME_SPIKE_MULTIPLIER
    
    MIN_AVG_VOLUME_20 = args.min_volume
    MIN_LAST_CLOSE = args.min_price
    BATCH_SIZE = args.batch
    DROP_THRESHOLD_PCT = args.drop_threshold
    LOOKBACK_BARS = args.lookback
    VOLUME_SPIKE_MULTIPLIER = args.volume_spike

    # Live intraday mode only
    if not args.live:
        print("❌ This script is for intraday alerts only. Use --live flag.")
        return

    # Email config
    smtp_user = args.smtp_user or os.environ.get("ALERT_FROM_EMAIL", "")
    to_list_raw = args.email_to or os.environ.get("ALERT_TO_EMAILS", "")
    to_list = [e.strip() for e in to_list_raw.split(",") if e.strip()]

    if not to_list:
        print("⚠️  --live requires recipients. Provide --email-to or set ALERT_TO_EMAILS in .stock_screener.env")
        return

    smtp_pass = os.environ.get(args.smtp_pass_env, "")

    print(f"🔄 Live sudden drop alerts: interval={args.interval}, period={args.period}, poll every {args.poll_secs}s")
    print(f"📧 Email to: {to_list} via {args.smtp_host}:{args.smtp_port} as {smtp_user or '(no user)'}")
    print(f"📉 Drop threshold: {DROP_THRESHOLD_PCT}% over {LOOKBACK_BARS} bars")

    LAST_ALERT_TS = {}   # debounce per ticker (epoch seconds)

    # Universe selection
    if args.universe.lower() == "nasdaq":
        tickers = get_nasdaq_composite_tickers()
    else:
        tickers = get_sp500_tickers()

    print(f"📊 Monitoring {len(tickers)} tickers for sudden drops...")

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
                                if len(batch) == 1:
                                    PRICE_DATA[batch[0]] = data.dropna().copy()
                                    
                    except Exception as e:
                        print(f"    ❌ Batch download error: {e}")
                        continue
                
                now_ts = time.time()
                alerts_sent = 0
                
                # Check each ticker for drops
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
                        
                        # Detect drop
                        drop_info = detect_sudden_drop(
                            tkr, 
                            DROP_THRESHOLD_PCT, 
                            LOOKBACK_BARS, 
                            VOLUME_SPIKE_MULTIPLIER,
                            require_volume_spike=args.require_volume_spike
                        )
                        
                        if drop_info is None:
                            continue
                        
                        # Debounce: configurable minutes per ticker
                        last_sent = LAST_ALERT_TS.get(tkr, 0)
                        if now_ts - last_sent >= args.debounce_minutes * 60:
                            # Company info
                            try:
                                company_name, sector, _ = get_company_info_robust(tkr, max_retries=1)
                            except Exception:
                                company_name, sector = tkr, "N/A"

                            subj = f"📉 SUDDEN DROP ALERT: {tkr} {company_name} — {args.interval}"
                            body = dedent(f"""\
                                SUDDEN PRICE DROP DETECTED!

                                Ticker:              {tkr}
                                Company:             {company_name}
                                Sector:              {sector}
                                Interval:            {args.interval}
                                Current Price:       ${drop_info['current_price']:.2f}
                                Previous Price:      ${drop_info['prev_price']:.2f}
                                Drop:                {drop_info['drop_pct']:.2f}% ({drop_info['drop_type']})
                                
                                Volume Information:
                                  Current Volume:    {drop_info['current_volume']:,.0f}
                                  Avg Volume (20d):  {drop_info['avg_volume']:,.0f}
                                  Volume Ratio:      {drop_info['volume_ratio']:.2f}x
                                
                                Price Context:
                                  52W High:          ${drop_info['high_52w']:.2f}
                                  52W Low:           ${drop_info['low_52w']:.2f}
                                  From 52W High:     {drop_info['price_from_52w_high']:.2f}%
                                  From 52W Low:      {drop_info['price_from_52w_low']:.2f}%
                                
                                Technical Indicators:
                                  RSI:               {drop_info['rsi']:.1f}
                                
                                Alert triggered at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            """)

                            try:
                                send_email_alert(
                                    args.smtp_host, args.smtp_port,
                                    smtp_user=smtp_user, smtp_pass=smtp_pass,
                                    to_list=to_list, subject=subj, body=body
                                )
                                LAST_ALERT_TS[tkr] = now_ts
                                alerts_sent += 1
                                print(f"📉 DROP ALERT SENT: {tkr} dropped {drop_info['drop_pct']:.2f}% to ${drop_info['current_price']:.2f}")
                            except Exception as ee:
                                print(f"⚠️  Email send failed for {tkr}: {ee}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {tkr}: {str(e)[:50]}...")
                        continue
                
                print(f"✅ Scan complete. {alerts_sent} alerts sent. Next scan in {args.poll_secs}s")
                
            else:
                print("⏸️  Market closed (PT) — sleeping…")

            time.sleep(max(15, int(args.poll_secs)))
            
    except KeyboardInterrupt:
        print("\n🛑 Live drop alerts stopped by user.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday sudden drop stock alerts.")
    
    # Basic CLI arguments
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (required for this script).")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME_20, help="Min 20-day avg volume.")
    parser.add_argument("--min-price", type=float, default=MIN_LAST_CLOSE, help="Min last close price.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size for yfinance download.")
    parser.add_argument("--universe", type=str, default="sp500", choices=["sp500", "nasdaq"],
                        help="Universe to scan (sp500 or nasdaq).")

    # Drop detection parameters
    parser.add_argument("--drop-threshold", type=float, default=DROP_THRESHOLD_PCT,
                        help="Percentage drop threshold to trigger alert (default: 10.0%%)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_BARS,
                        help="Number of bars to look back for drop calculation (default: 5)")
    parser.add_argument("--volume-spike", type=float, default=VOLUME_SPIKE_MULTIPLIER,
                        help="Volume spike multiplier threshold (default: 1.5x)")
    parser.add_argument("--require-volume-spike", action="store_true",
                        help="Require volume spike along with price drop (optional)")
    parser.add_argument("--debounce-minutes", type=int, default=30,
                        help="Minutes between alerts for same ticker (default: 30)")

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