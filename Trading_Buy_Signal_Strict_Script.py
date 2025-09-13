# file: stock_multi_method_screener.py
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
import os
import time
import json
import gzip

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
# Universe
# =========================
def get_sp500_tickers(use_cache=True, cache_path="sp500_tickers_cache.json.gz", max_cache_age_hours=12):
    """
    Return a list of S&P 500 ticker symbols (normalized for yfinance).
    Strategy: try multiple data sources with graceful fallback.
    - Primary: GitHub dataset (CSV)
    - Secondary: StockAnalysis.com list (HTML table)
    - Tertiary: Wikipedia (HTML table, with headers)
    Caches results locally to avoid rate limits and transient errors.
    """
    def _normalize(tickers):
        return (
            pd.Series(tickers)
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)  # e.g., BRK.B -> BRK-B
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )

    def _valid(tickers):
        # S&P 500 has ~500 constituents; allow a range for transient changes
        return isinstance(tickers, list) and 450 <= len(tickers) <= 520

    # ---------- Optional cache ----------
    if use_cache and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".gz"):
                with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            age_hours = (time.time() - payload["timestamp"]) / 3600.0
            if _valid(payload["tickers"]) and age_hours <= max_cache_age_hours:
                return payload["tickers"]
        except Exception:
            pass  # ignore cache errors

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    })
    timeout = 20

    errors = []

    # ---------- Source 1: GitHub CSV (datasets/s-and-p-500-companies) ----------
    # Maintained dataset with a raw CSV including a 'Symbol' column.
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "Symbol" in df.columns:
            tickers = _normalize(df["Symbol"])
            if _valid(tickers):
                # cache
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                except Exception:
                    pass
                return tickers
        else:
            errors.append("GitHub CSV missing 'Symbol' column")
    except Exception as e:
        errors.append(f"GitHub CSV failed: {e}")

    # ---------- Source 2: StockAnalysis.com table ----------
    # Public list page renders a clean table that pandas can read.
    try:
        url = "https://stockanalysis.com/list/sp-500-stocks/"
        tables = pd.read_html(session.get(url, timeout=timeout).text)
        # find table with a Symbol column
        df = next((t for t in tables if any(c.lower() == "symbol" for c in map(str, t.columns))), None)
        if df is not None:
            # Column could be named exactly 'Symbol' or similar
            symbol_col = next(c for c in df.columns if str(c).lower() == "symbol")
            tickers = _normalize(df[symbol_col])
            if _valid(tickers):
                # cache
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                except Exception:
                    pass
                return tickers
        else:
            errors.append("StockAnalysis page: no table with Symbol found")
    except Exception as e:
        errors.append(f"StockAnalysis failed: {e}")

    # ---------- Source 3: Wikipedia (with a proper UA) ----------
    # The page layout changes; we scan tables for the constituents.
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = session.get(url, timeout=timeout).text
        tables = pd.read_html(html)
        sp500 = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if "symbol" in cols and any(k in cols for k in ["security", "company", "name"]):
                sp500 = t
                break
        if sp500 is not None:
            symbol_col = next(c for c in sp500.columns if str(c).strip().lower() == "symbol")
            tickers = _normalize(sp500[symbol_col])
            if _valid(tickers):
                # cache
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                except Exception:
                    pass
                return tickers
        else:
            errors.append("Wikipedia: constituents table not found")
    except Exception as e:
        errors.append(f"Wikipedia failed: {e}")

    # ---------- Final fallback ----------
    # If everything fails, return a minimal, high-cap set so the rest of your pipeline still runs.
    print("Warning: all sources failed. Errors:", " | ".join(errors))
    return ["AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "NVDA", "BRK-B", "UNH", "XOM"]

def get_nasdaq_composite_tickers():
    """
    Fetch Nasdaq-listed common stocks from Nasdaq Trader (HTTPS),
    filtering out ETFs, test issues, NextShares, etc.
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep="|")

        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["NextShares"] == "N")]
        tickers = df["Symbol"].dropna().tolist()
        print(f"Found {len(tickers)} Nasdaq-listed common stocks")
        return tickers
    except Exception as e:
        print(f"Error fetching Nasdaq list: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

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
            
            # Initialize defaults
            company_name = ticker  # Default to ticker
            sector = "N/A"
            market_cap = np.nan
            
            # Try fast_info first (faster but less reliable)
            fi = getattr(ti, "fast_info", None)
            if fi:
                # Extract company name from fast_info
                company_name = (getattr(fi, "long_name", None) or 
                              getattr(fi, "short_name", None) or 
                              getattr(fi, "display_name", None) or
                              getattr(fi, "name", None))
                
                # Extract sector and market cap
                sector = getattr(fi, "sector", "N/A") or "N/A"
                market_cap = getattr(fi, "market_cap", np.nan)
                
                # Clean up company name
                if company_name and company_name.strip() and company_name != "N/A":
                    company_name = company_name.strip()
                else:
                    company_name = None  # Mark for .info fallback
            
            # If fast_info failed or returned empty, try .info (slower but more complete)
            if not company_name or company_name == ticker or sector == "N/A":
                try:
                    info = ti.info
                    
                    # Get company name from .info if we don't have a good one
                    if not company_name or company_name == ticker:
                        info_name = (info.get('longName') or 
                                   info.get('shortName') or 
                                   info.get('displayName'))
                        if info_name and info_name.strip():
                            company_name = info_name.strip()
                    
                    # Get sector from .info if we don't have it
                    if sector == "N/A":
                        info_sector = info.get('sector') or info.get('industry')
                        if info_sector and info_sector.strip():
                            sector = info_sector.strip()
                    
                    # Get market cap from .info if we don't have it
                    if pd.isna(market_cap):
                        info_mcap = info.get('marketCap')
                        if info_mcap:
                            market_cap = float(info_mcap)
                            
                except Exception as info_error:
                    # .info failed, but we might have partial data from fast_info
                    pass
            
            # Ensure we have a company name (fallback to ticker if all else fails)
            if not company_name or not company_name.strip():
                company_name = ticker
            
            # Return successful result
            return company_name, sector, market_cap
                
        except Exception as e:
            # If this isn't our last attempt, wait and retry
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed for {ticker}: {str(e)[:50]}... Retrying...")
                time.sleep(0.5 * (attempt + 1))  # Progressive delay: 0.5s, 1.0s, 1.5s
                continue
            else:
                # Final attempt failed
                print(f"  ❌ All attempts failed for {ticker}. Using ticker as company name.")
    
    # All attempts failed - return minimal fallback data
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

#initial version with small fixes
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
# Recent signal scan
# =========================
def check_recent_multi_buy_signals(df, ticker, lookback_days=7):
    """
    Check if there are buy signals from ALL methodologies in the last N trading bars.
    (NEW) Require SMA200 uptrend + MACD confirmation. Include ATR-based stops in output.
    """
    if df is None or df.empty:
        return None

    recent_df = df.tail(lookback_days)
    if recent_df.empty:
        return None

    sma_signals = int(recent_df['SMA_Buy_Signal'].sum()) + int((recent_df['SMA20'] > recent_df['SMA50']).iloc[-1])
    ema_rsi_signals = int(recent_df['EMA_RSI_Buy_Signal'].sum())
    bb_signals = int(recent_df['BB_Buy_Signal'].sum())
    all_methods_signals = int(recent_df['All_Methods_Buy'].sum())
    quality_signals = int(recent_df['Quality_Buy_Signal'].sum())

    # (NEW) Trend & MACD filters
    latest_close = float(df['Close'].iloc[-1])
    sma200_val = float(df['SMA200'].iloc[-1]) if 'SMA200' in df.columns and pd.notna(df['SMA200'].iloc[-1]) else np.nan
    macd_line_now = float(df['MACD_Line'].iloc[-1]) if 'MACD_Line' in df.columns else np.nan
    macd_signal_now = float(df['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in df.columns else np.nan

    trend_ok = (not np.isnan(sma200_val)) and (latest_close > sma200_val)
    macd_bullish_now = (not np.isnan(macd_line_now)) and (not np.isnan(macd_signal_now)) and (macd_line_now > macd_signal_now)

    macd_cross_up_recent = False
    if {'MACD_Line','MACD_Signal'}.issubset(recent_df.columns):
        macd_cross_up_recent = ((recent_df['MACD_Line'] > recent_df['MACD_Signal']) &
                                (recent_df['MACD_Line'].shift(1) <= recent_df['MACD_Signal'].shift(1))).any()

    # Only accept if base methods fired AND SMA200 uptrend AND MACD bullish (now or recent cross)
    if all_methods_signals > 0 and trend_ok and (macd_bullish_now or macd_cross_up_recent):
        all_signal_dates = recent_df[recent_df['All_Methods_Buy']].index
        last_all_signal = all_signal_dates.max() if len(all_signal_dates) > 0 else None

        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_ema5 = float(df['EMA5'].iloc[-1])
        current_ema20 = float(df['EMA20'].iloc[-1])
        current_sma20 = float(df['SMA20'].iloc[-1])
        current_sma50 = float(df['SMA50'].iloc[-1])

        ema_strength = ((current_ema5 - current_ema20) / current_ema20) * 100 if current_ema20 else 0.0
        sma_strength = ((current_sma20 - current_sma50) / current_sma50) * 100 if current_sma50 else 0.0
        bb_position = ((current_price - df['BB_Lower'].iloc[-1]) / df['BB_Range'].iloc[-1]) * 100

        # (NEW) ATR-based stops
        atr14 = float(df['ATR14'].iloc[-1]) if 'ATR14' in df.columns and pd.notna(df['ATR14'].iloc[-1]) else np.nan
        atr_stop_1_5x = current_price - 1.5 * atr14 if not np.isnan(atr14) else np.nan
        atr_stop_2x   = current_price - 2.0 * atr14 if not np.isnan(atr14) else np.nan

        # Avoid slow .info; fast_info is OK for mcap
           # Enhanced company info extraction with retries
        try:
            company_name, sector, market_cap = get_company_info_robust(ticker, max_retries=2)
        except Exception as e:
            print(f"  ❌ Company info extraction completely failed for {ticker}: {e}")
            company_name = ticker
            sector = "N/A" 
            market_cap = np.nan

        return {
            'Ticker': ticker,
            'Company': company_name,
            'Sector': sector,
            'Current_Price': current_price,
            'RSI': current_rsi,
            'EMA_Strength_%': ema_strength,
            'SMA_Strength_%': sma_strength,
            'BB_Position_%': bb_position,
            'SMA_Signals': int(sma_signals > 0),
            'EMA_RSI_Signals': ema_rsi_signals,
            'BB_Signals': bb_signals,
            'All_Methods_Signals': all_methods_signals,
            'Quality_Signals': quality_signals,
            'Last_Signal_Date': last_all_signal,
            'Market_Cap': market_cap,
            # (NEW) qualifiers & stops
            'Above_SMA200': int(trend_ok),
            'MACD_Bullish': int(macd_bullish_now or macd_cross_up_recent),
            'ATR14': atr14,
            'ATR_Stop_1_5x': atr_stop_1_5x,
            'ATR_Stop_2x': atr_stop_2x
        }

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
# Main screening (chunked)
# =========================
def screen_stocks_multi_methodology(
    min_avg_volume_20=MIN_AVG_VOLUME_20,
    min_last_close=MIN_LAST_CLOSE,
    batch_size=BATCH_SIZE,
    lb_days=lookback_days,
    _tickers_override=None,
    max_cache_size=500  # NEW: Prevent memory issues
):
    """
    Screen stocks for buy signals from ALL methodologies.
    Uses fast chunked downloads and a global PRICE_DATA cache.
    """
    tickers = _tickers_override or get_sp500_tickers()
    n_tickers = len(tickers)
    multi_buy_signal_stocks = []

    PRICE_DATA.clear()

    print(f"\nPrefetching historical data in batches of {batch_size}...")
    
    # Process in chunks to manage memory
    for chunk_start in range(0, n_tickers, max_cache_size):
        chunk_end = min(chunk_start + max_cache_size, n_tickers)
        chunk_tickers = tickers[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//max_cache_size + 1}: tickers {chunk_start}-{chunk_end}")
        
        # Clear cache for each chunk to manage memory
        if chunk_start > 0:
            PRICE_DATA.clear()
        
        # Batch download for this chunk
        for i, batch in enumerate(_chunks(chunk_tickers, batch_size), start=1):
            try:
                data = yf.download(batch, start=start_date, end=end_date, progress=False,
                                   group_by='ticker', threads=True)
                
                # Always handle as MultiIndex (more robust)
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        for tkr in batch:
                            try:
                                sub = _slice_from_batch(data, tkr)
                                if sub is not None and not sub.empty:
                                    PRICE_DATA[tkr] = sub
                            except Exception as e:
                                print(f"  ⚠️  Failed to extract {tkr}: {e}")
                    else:
                        # Single ticker case
                        if len(batch) == 1:
                            PRICE_DATA[batch[0]] = data.dropna().copy()
                        
            except Exception as e:
                print(f"Batch download error: {e}")

        # Apply liquidity filter and screen this chunk
        filtered_chunk = []
        for tkr in chunk_tickers:
            try:
                df = PRICE_DATA.get(tkr)
                if df is None or df.empty:
                    continue
                    
                # Liquidity check using cached data
                if len(df) < 20:
                    continue
                    
                vol_ok = (min_avg_volume_20 <= 0 or 
                         float(df['Volume'].tail(20).mean()) >= float(min_avg_volume_20))
                px_ok = (min_last_close <= 0 or 
                        float(df['Close'].iloc[-1]) >= float(min_last_close))
                
                if vol_ok and px_ok:
                    filtered_chunk.append(tkr)
                    
            except Exception as e:
                print(f"  ⚠️  Liquidity filter error for {tkr}: {e}")
        
        print(f"Chunk liquidity filter: {len(filtered_chunk)}/{len(chunk_tickers)} tickers")
        
        # Screen this chunk
        for i, ticker in enumerate(filtered_chunk, start=1):
            try:
                if (i % 50) == 0:
                    print(f"  Processed {i}/{len(filtered_chunk)} in current chunk...")

                df = calculate_indicators(ticker, start_date, end_date)
                result = check_recent_multi_buy_signals(df, ticker, lb_days)

                if result:
                    multi_buy_signal_stocks.append(result)
                    print(f"✅ {ticker}: Multi-methodology buy signal found!")
                    
            except Exception as e:
                print(f"❌ Error processing {ticker}: {str(e)[:100]}...")
                continue

    return multi_buy_signal_stocks, n_tickers

# =========================
# Pretty console output
# =========================
def _print_console(results, total_screened):
    print("\n" + "=" * 80)
    print("🎯 SP500 STOCKS WITH BUY SIGNALS FROM ALL METHODOLOGIES")
    print("=" * 80)

    if results:
        results_df = pd.DataFrame(results)
        results_df['Combined_Strength'] = results_df['EMA_Strength_%'] + results_df['SMA_Strength_%']
        results_df = results_df.sort_values('Combined_Strength', ascending=False)

        print(f"\n📊 SUMMARY:")
        print(f"Total stocks screened: {total_screened}")
        print(f"Stocks with ALL methodology buy signals: {len(results_df)}")
        success_rate = (len(results_df) / max(total_screened, 1)) * 100
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Quality signals (with volume): {int(results_df['Quality_Signals'].sum())}")

        print(f"\n📈 DETAILED RESULTS:")
        print("-" * 140)

        display_cols = ['Ticker', 'Company', 'Sector', 'Current_Price', 'RSI',
                        'EMA_Strength_%', 'SMA_Strength_%', 'BB_Position_%',
                        'All_Methods_Signals', 'Quality_Signals', 'Last_Signal_Date',
                        # NEW fields appended
                        'Above_SMA200', 'MACD_Bullish', 'ATR14', 'ATR_Stop_1_5x', 'ATR_Stop_2x']
        existing_cols = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[existing_cols].copy()

        display_df['Current_Price'] = display_df['Current_Price'].map(lambda x: f"${x:.2f}")
        display_df['RSI'] = display_df['RSI'].map(lambda x: f"{x:.1f}")
        display_df['EMA_Strength_%'] = display_df['EMA_Strength_%'].map(lambda x: f"{x:.2f}%")
        display_df['SMA_Strength_%'] = display_df['SMA_Strength_%'].map(lambda x: f"{x:.2f}%")
        display_df['BB_Position_%'] = display_df['BB_Position_%'].map(lambda x: f"{x:.1f}%")
        display_df['Last_Signal_Date'] = display_df['Last_Signal_Date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 25)
        print(display_df.to_string(index=False))

        print(f"\n📊 METHODOLOGY VALIDATION:")
        print(f"Stocks with SMA signals: {int(results_df['SMA_Signals'].sum())}")
        print(f"Stocks with EMA+RSI signals: {int(results_df['EMA_RSI_Signals'].sum())}")
        print(f"Stocks with Bollinger Bands signals: {int(results_df['BB_Signals'].sum())}")
        print(f"Stocks with ALL methodologies: {len(results_df)}")

        print(f"\n📊 SECTOR BREAKDOWN:")
        sector_counts = results_df['Sector'].value_counts()
        for sector, count in sector_counts.items():
            print(f"{sector}: {count} stocks")

        print(f"\n🏆 TOP 10 STRONGEST MULTI-METHODOLOGY SIGNALS:")
        top_10 = results_df.head(10)
        for _, row in top_10.iterrows():
            print(f"{row['Ticker']} ({str(row['Company'])[:25]}...)")
            print(f"   EMA: {row['EMA_Strength_%']:.2f}%, SMA: {row['SMA_Strength_%']:.2f}%, RSI: {row['RSI']:.1f}, BB: {row['BB_Position_%']:.1f}%")

        quality_stocks = results_df[results_df['Quality_Signals'] > 0]
        if len(quality_stocks) > 0:
            print(f"\n⭐ HIGH-QUALITY SIGNALS (with volume confirmation): {len(quality_stocks)} stocks")
            for _, row in quality_stocks.iterrows():
                print(f"   {row['Ticker']} - {str(row['Company'])[:30]}")
    else:
        print("❌ No stocks found with buy signals from ALL methodologies in the last window.")
        print("This is normal - the multi-methodology approach is very selective!")
        print("Consider:\n- Extending the lookback period (try 14 days)\n- Running during different market conditions\n- Checking individual methodology results")
    print("\n" + "=" * 80)
    print("🏁 Multi-methodology screening completed!")
    print("=" * 80)

# =========================
# CLI entrypoint
# =========================
def run_cli(args):
    global MIN_AVG_VOLUME_20, MIN_LAST_CLOSE, lookback_days, BATCH_SIZE
    MIN_AVG_VOLUME_20 = args.min_volume
    MIN_LAST_CLOSE = args.min_price
    lookback_days = args.lookback
    BATCH_SIZE = args.batch

    # Universe selection
    if args.universe.lower() == "nasdaq":
        tickers = get_nasdaq_composite_tickers()
    else:
        tickers = get_sp500_tickers()

    results, total_screened = screen_stocks_multi_methodology(
        min_avg_volume_20=MIN_AVG_VOLUME_20,
        min_last_close=MIN_LAST_CLOSE,
        batch_size=BATCH_SIZE,
        lb_days=lookback_days,
        _tickers_override=tickers
    )
    _print_console(results, total_screened)

# =========================
# Streamlit app (optional)
# =========================
def run_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    from datetime import timedelta

    st.set_page_config(page_title="Stocks Multi-Method Screener", layout="wide")
    st.title("📈 SP500 / Nasdaq Multi-Methodology Strict Screener")
    st.caption("SMA crossover • EMA+RSI • Bollinger bounce/cross • Liquidity filter")

    with st.sidebar:
        st.subheader("Scan settings")
        universe = st.selectbox("Universe", options=["S&P 500", "Nasdaq Composite"], index=0)
        min_vol = st.number_input("Min 20-day avg volume", value=int(MIN_AVG_VOLUME_20), step=50_000, min_value=0)
        min_px = st.number_input("Min last close ($)", value=float(MIN_LAST_CLOSE), step=0.5, min_value=0.0, format="%.2f")
        lb_days = st.number_input("Lookback bars", value=int(lookback_days), step=1, min_value=3, max_value=30)
        batch = st.number_input("Batch size", value=int(BATCH_SIZE), step=20, min_value=20, max_value=200)

        run_btn = st.button("Run scan now")
        st.caption(f"Window: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

    @st.cache_data(ttl=24*3600, show_spinner=False)
    def run_scan_cached(universe, min_vol, min_px, batch, lb_days):
        PRICE_DATA.clear()
        # Decide universe here; pass via override to keep structure
        if universe == "Nasdaq Composite":
            tickers = get_nasdaq_composite_tickers()
        else:
            tickers = get_sp500_tickers()

        results, total = screen_stocks_multi_methodology(
            min_avg_volume_20=min_vol,
            min_last_close=min_px,
            batch_size=batch,
            lb_days=lb_days,
            _tickers_override=tickers
        )
        df = pd.DataFrame(results) if results else pd.DataFrame()
        if not df.empty:
            df["Combined_Strength"] = df["EMA_Strength_%"] + df["SMA_Strength_%"]
            df = df.sort_values("Combined_Strength", ascending=False)
        return df, total

    if run_btn:
        st.toast("Running fresh scan… this can take several minutes.", icon="⏳")
        st.cache_data.clear()

    results_df, total_screened = run_scan_cached(universe, min_vol, min_px, batch, lb_days)

    if results_df.empty:
        st.info("No stocks found with buy signals from ALL methodologies in the last window.")
        return

    quality_signals = int(results_df["Quality_Signals"].sum()) if "Quality_Signals" in results_df else 0
    success_rate = 100.0 * len(results_df) / max(total_screened, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe", universe)
    m2.metric("Total screened", f"{total_screened:,}")
    m3.metric("Multi-method signals", f"{len(results_df):,}")
    m4.metric("Hit rate", f"{success_rate:.2f}%")

    st.subheader("🏆 Top 10 by Combined Strength")
    top10 = results_df.head(10)
    if not top10.empty:
        st.table(top10[['Ticker', 'Company', 'EMA_Strength_%', 'SMA_Strength_%', 'RSI', 'BB_Position_%']])

    st.subheader("📈 Multi-Methodology Signal Plots")
    plot_ticker = st.selectbox(
        "Select ticker to plot:",
        options=results_df['Ticker'].tolist(),
        index=0
    )

    if plot_ticker:
        try:
            company = results_df.loc[results_df['Ticker'] == plot_ticker, 'Company'].iloc[0]
            if str(company).strip().upper() == str(plot_ticker).strip().upper() or not str(company).strip():
                try:
                    info = yf.Ticker(plot_ticker).info
                    company = info.get('longName') or info.get('shortName') or company
                except Exception:
                    pass

            # --- compute long so SMA200/MACD are defined; show ~6 months ---
            compute_start = end_date - timedelta(days=420)   # ~14 months
            plot_cutoff   = end_date - timedelta(days=185)   # ~6 months visible

            raw = yf.download(plot_ticker, start=compute_start, end=end_date, progress=False)
            if raw.empty:
                st.error(f"Could not download data for {plot_ticker}")
            else:
                # Robust extraction (handles MultiIndex)
                if isinstance(raw.columns, pd.MultiIndex):
                    try:
                        close_col = raw[('Close', plot_ticker)]
                        high_col  = raw[('High',  plot_ticker)]
                        low_col   = raw[('Low',   plot_ticker)]
                        vol_col   = raw[('Volume',plot_ticker)]
                    except KeyError:
                        sub = raw.xs(plot_ticker, axis=1, level=1, drop_level=False)
                        sub.columns = sub.columns.get_level_values(0)
                        close_col = sub['Close']; high_col = sub['High']; low_col = sub['Low']; vol_col = sub['Volume']
                else:
                    close_col = raw['Close']; high_col = raw['High']; low_col = raw['Low']; vol_col = raw['Volume']

                df = pd.DataFrame({
                    'Close': close_col.astype(float),
                    'High' : high_col.astype(float),
                    'Low'  : low_col.astype(float),
                    'Volume': vol_col.astype(float)
                })

                # ========= Indicators (exactly like your scanner) =========
                # SMA20/50 + crossover
                df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
                df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
                df["SMA_Buy_Signal"]  = (df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
                df["SMA_Sell_Signal"] = (df["SMA20"] < df["SMA50"]) & (df["SMA20"].shift(1) >= df["SMA50"].shift(1))

                # EMA/RSI (strict inequalities)
                df["EMA5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
                df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
                delta = df["Close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14, min_periods=14).mean()
                avg_loss = loss.rolling(14, min_periods=14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df["RSI"] = 100 - (100 / (1 + rs))
                df["RSI"] = df["RSI"].fillna(50)
                df["EMA_RSI_Buy_Signal"] = (df["EMA5"] > df["EMA20"]) & (df["RSI"] > 50) & (df["RSI"] < 70)

                # Bollinger (buy/sell like your plot)
                df['BB_Middle'] = df["Close"].rolling(20, min_periods=20).mean()
                bb_std = df["Close"].rolling(20, min_periods=20).std()
                df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
                df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
                df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

                bb_lower_bounce   = (df["Close"].shift(1) <= df['BB_Lower'].shift(1) * 1.02) & (df["Close"] > df['BB_Lower'] * 1.02)
                bb_middle_cross   = (df["Close"] > df['BB_Middle']) & (df["Close"].shift(1) <= df['BB_Middle'].shift(1))
                bb_width_ok       = df['BB_Width'] > df['BB_Width'].rolling(20, min_periods=20).mean() * 0.8
                df["BB_Buy_Signal"]  = (bb_lower_bounce | bb_middle_cross) & bb_width_ok

                bb_upper_bounce      = (df["Close"].shift(1) >= df['BB_Upper'].shift(1) * 0.98) & (df["Close"] < df['BB_Upper'] * 0.98)
                bb_middle_cross_down = (df["Close"] < df['BB_Middle']) & (df["Close"].shift(1) >= df['BB_Middle'].shift(1))
                df["BB_Sell_Signal"] = (bb_upper_bounce | bb_middle_cross_down) & bb_width_ok

                # Base multi-method combos (same as scanner)
                df["All_Methods_Buy"]  = ((df["SMA_Buy_Signal"] | (df["SMA20"] > df["SMA50"])) &
                                        df["EMA_RSI_Buy_Signal"] & df["BB_Buy_Signal"])
                df["All_Methods_Sell"] = ((df["SMA_Sell_Signal"] | (df["SMA20"] < df["SMA50"])) &
                                        (df["EMA5"] < df["EMA20"]) & ~((df["RSI"] > 50) & (df["RSI"] < 70)) & df["BB_Sell_Signal"])

                # Trend & MACD (strict gate components)
                df["SMA200"] = df["Close"].rolling(200, min_periods=200).mean()
                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                ema26 = df["Close"].ewm(span=26, adjust=False).mean()
                df["MACD_Line"]   = ema12 - ema26
                df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()

                # === Mirror scanner acceptance timing ===
                # Look back N bars for a base buy
                N = int(lookback_days) if isinstance(lookback_days, int) else 5
                recent = df.tail(N)

                # Gate evaluated "now" (or recent MACD cross up)
                trend_ok_now      = (df["Close"].iloc[-1] > df["SMA200"].iloc[-1]) if pd.notna(df["SMA200"].iloc[-1]) else False
                macd_bull_now     = (df["MACD_Line"].iloc[-1] > df["MACD_Signal"].iloc[-1]) if pd.notna(df["MACD_Line"].iloc[-1]) and pd.notna(df["MACD_Signal"].iloc[-1]) else False
                macd_cross_up_recent = (((recent["MACD_Line"] > recent["MACD_Signal"]) &
                                        (recent["MACD_Line"].shift(1) <= recent["MACD_Signal"].shift(1))).any()
                                        if {'MACD_Line','MACD_Signal'}.issubset(recent.columns) else False)

                # Last base buy within lookback
                last_base_buy_idx = recent.index[recent["All_Methods_Buy"].fillna(False)].max() if recent["All_Methods_Buy"].fillna(False).any() else None

                # Strict SELL (symmetric definition; optional)
                macd_bear_now        = (df["MACD_Line"].iloc[-1] < df["MACD_Signal"].iloc[-1]) if pd.notna(df["MACD_Line"].iloc[-1]) and pd.notna(df["MACD_Signal"].iloc[-1]) else False
                macd_cross_down_recent = (((recent["MACD_Line"] < recent["MACD_Signal"]) &
                                        (recent["MACD_Line"].shift(1) >= recent["MACD_Signal"].shift(1))).any()
                                        if {'MACD_Line','MACD_Signal'}.issubset(recent.columns) else False)
                last_base_sell_idx = recent.index[recent["All_Methods_Sell"].fillna(False)].max() if recent["All_Methods_Sell"].fillna(False).any() else None
                downtrend_now = (df["Close"].iloc[-1] < df["SMA200"].iloc[-1]) if pd.notna(df["SMA200"].iloc[-1]) else False

                # Slice for plotting AFTER we find indices (so NaNs earlier don’t drop the event)
                df_plot = df.loc[df.index >= plot_cutoff].copy()

                # ===== Plot (strict markers placed at the base-event date) =====
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

                ax1.plot(df_plot.index, df_plot["Close"], label="Close", linewidth=1.2, alpha=0.9)
                ax1.plot(df_plot.index, df_plot["SMA20"], label="SMA20", linewidth=1.0)
                ax1.plot(df_plot.index, df_plot["SMA50"], label="SMA50", linewidth=1.0)
                ax1.plot(df_plot.index, df_plot["SMA200"], label="SMA200 (gate)", linewidth=1.0)

                # --- NEW: show historical base multi-method signals on the chart (ungated) ---
                hist_buy  = df_plot[df_plot["All_Methods_Buy"].fillna(False)]
                hist_sell = df_plot[df_plot["All_Methods_Sell"].fillna(False)]
                if not hist_buy.empty:
                    ax1.scatter(hist_buy.index,  hist_buy["Close"],  marker="^", s=60, alpha=0.45,
                                label="Multi-Method BUY (history)", zorder=4)
                if not hist_sell.empty:
                    ax1.scatter(hist_sell.index, hist_sell["Close"], marker="v", s=60, alpha=0.45,
                                label="Multi-Method SELL (history)", zorder=4)
                # ---------------------------------------------------------------------------

                # Strict BUY marker if scanner would accept now
                if last_base_buy_idx is not None and trend_ok_now and (macd_bull_now or macd_cross_up_recent):
                    if last_base_buy_idx in df_plot.index:
                        ax1.scatter([last_base_buy_idx], [df_plot.loc[last_base_buy_idx, "Close"]],
                                    marker="^", s=140, zorder=6, label="Multi-Method BUY (STRICT)")

                # Strict SELL marker (optional symmetric logic)
                if last_base_sell_idx is not None and downtrend_now and (macd_bear_now or macd_cross_down_recent):
                    if last_base_sell_idx in df_plot.index:
                        ax1.scatter([last_base_sell_idx], [df_plot.loc[last_base_sell_idx, "Close"]],
                                    marker="v", s=140, zorder=6, label="Multi-Method SELL (STRICT)")

                ax1.set_title(f"{plot_ticker} {company} — Strict Multi-Method Signals (last ~6 months)", fontsize=14, fontweight='bold')
                ax1.set_ylabel("Price (USD)")
                ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                ax1.grid(True, alpha=0.3)

                # RSI panel
                ax2.plot(df_plot.index, df_plot["RSI"], label="RSI", linewidth=1.2)
                ax2.axhline(70, linestyle='--', alpha=0.7, label='Overbought (70)')
                ax2.axhline(30, linestyle='--', alpha=0.7, label='Oversold (30)')
                ax2.axhline(50, linestyle='-',  alpha=0.5, label='Neutral (50)')
                ax2.set_ylabel("RSI")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # MACD panel
                ax3.plot(df_plot.index, df_plot["MACD_Line"],   label="MACD line", linewidth=1.2)
                ax3.plot(df_plot.index, df_plot["MACD_Signal"], label="Signal",    linewidth=1.0)
                ax3.axhline(0, linestyle='--', alpha=0.6)
                ax3.set_ylabel("MACD")
                ax3.set_xlabel("Date")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # quick visibility counts
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Base buys in lookback", int(recent["All_Methods_Buy"].fillna(False).sum()))
                with col2:
                    st.metric("Gate now (SMA200 & MACD)", int(trend_ok_now and (macd_bull_now or macd_cross_up_recent)))
                with col3:
                    st.metric("Days plotted", len(df_plot))

        except Exception as e:
            st.error(f"Error plotting {plot_ticker}: {str(e)}")



    st.subheader("📊 Methodology Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Top 10 by EMA Strength**")
        top_ema = results_df.nlargest(10, 'EMA_Strength_%')[['Ticker', 'Company', 'EMA_Strength_%', 'RSI']]
        st.dataframe(top_ema, use_container_width=True, height=300)

    with col2:
        st.write("**Top 10 by SMA Strength**")
        top_sma = results_df.nlargest(10, 'SMA_Strength_%')[['Ticker', 'Company', 'SMA_Strength_%', 'BB_Position_%']]
        st.dataframe(top_sma, use_container_width=True, height=300)

    with col3:
        st.write("**Top 10 by Combined Strength**")
        top_combined = results_df.head(10)[['Ticker', 'Company', 'EMA_Strength_%', 'SMA_Strength_%', 'Combined_Strength']]
        st.dataframe(top_combined, use_container_width=True, height=300)

    st.subheader("📋 Complete Results")
    display_cols = ['Ticker', 'Company', 'Sector', 'Current_Price', 'RSI',
                    'EMA_Strength_%', 'SMA_Strength_%', 'BB_Position_%',
                    'All_Methods_Signals', 'Quality_Signals', 'Last_Signal_Date',
                    # NEW fields appended to table
                    'Above_SMA200', 'MACD_Bullish', 'ATR14', 'ATR_Stop_1_5x', 'ATR_Stop_2x']
    existing_cols = [c for c in display_cols if c in results_df.columns]
    show_df = results_df[existing_cols].copy()

    if 'Current_Price' in show_df:
        show_df['Current_Price'] = show_df['Current_Price'].map(lambda x: f"${x:.2f}")
    if 'RSI' in show_df:
        show_df['RSI'] = show_df['RSI'].map(lambda x: f"{x:.1f}")
    if 'EMA_Strength_%' in show_df:
        show_df['EMA_Strength_%'] = show_df['EMA_Strength_%'].map(lambda x: f"{x:.2f}%")
    if 'SMA_Strength_%' in show_df:
        show_df['SMA_Strength_%'] = show_df['SMA_Strength_%'].map(lambda x: f"{x:.2f}%")
    if 'BB_Position_%' in show_df:
        show_df['BB_Position_%'] = show_df['BB_Position_%'].map(lambda x: f"{x:.1f}%")
    if 'Last_Signal_Date' in show_df:
        show_df['Last_Signal_Date'] = pd.to_datetime(show_df['Last_Signal_Date']).dt.strftime('%Y-%m-%d')

    st.dataframe(show_df, use_container_width=True, height=520)

    st.download_button(
        "Download full CSV",
        data=results_df.to_csv(index=False),
        file_name="stocks_multi_methodology_buy_signals.csv",
        mime="text/csv",
        use_container_width=True,
    )

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SP500 multi-method screener (CLI or Streamlit).")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (default is Streamlit).")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME_20, help="Min 20-day avg volume.")
    parser.add_argument("--min-price", type=float, default=MIN_LAST_CLOSE, help="Min last close price.")
    parser.add_argument("--lookback", type=int, default=lookback_days, help="Lookback bars for signals.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size for yfinance download.")
    parser.add_argument("--universe", type=str, default="sp500", choices=["sp500", "nasdaq"],
                        help="Universe to scan (sp500 or nasdaq).")
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        # Launch Streamlit app when executed normally:
        #   streamlit run sp500_multi_method_screener.py
        run_streamlit()
