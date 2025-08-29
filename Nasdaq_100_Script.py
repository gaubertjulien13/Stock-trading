# file: nasdaq_100_multi_method_screener.py
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

# =========================
# Global config / cache
# =========================
PRICE_DATA = {}                 # Batch cache
BATCH_SIZE = 60                 # Tune for rate limits
MIN_AVG_VOLUME_20 = 150_000     # 0 to disable
MIN_LAST_CLOSE = 3.0            # 0 to disable

# Default time window
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
lookback_days = 3  # bars, not calendar days

# =========================
# Helper: chunking
# =========================
def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# Universe
# =========================
def get_nasdaq_tickers():
    """Get NASDAQ 100 ticker symbols from Wikipedia"""
    try:
        # Read NASDAQ 100 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        nasdaq_table = tables[4]  # The table with company list is usually the 5th table (index 4)
        tickers = nasdaq_table['Ticker'].tolist()
        
        # Clean up ticker symbols (some may have periods)
        tickers = [ticker.replace('.', '-') for ticker in tickers if isinstance(ticker, str)]
        
        print(f"Found {len(tickers)} NASDAQ tickers")
        return tickers
    except Exception as e:
        print(f"Error fetching NASDAQ list: {e}")
        # Fallback to a sample of major NASDAQ stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'INTC',
                'CSCO', 'PYPL', 'CMCSA', 'PEP', 'COST', 'AVGO', 'TXN', 'QCOM', 'TMUS', 'AMGN']
# =========================
# Indicators
# =========================
def _extract_cols(df, ticker=None):
    """
    Return (close, high, low, volume) Series regardless of single or MultiIndex columns.
    After batch caching, per-ticker frames are single-level (fields only).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Handle either orientation robustly
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker is not None and ticker in lv0:
            sub = df[ticker]
        elif ticker is not None and ticker in lv1:
            sub = df.xs(ticker, axis=1, level=1)
        else:
            # Fallback: if either level looks like fields, pick the other as ticker
            if set(['Close','Open','High','Low','Volume']).issubset(set(lv0)):
                sub = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
            else:
                sub = df.xs(df.columns.get_level_values(0)[0], axis=1, level=0)
        if isinstance(sub, pd.DataFrame) and isinstance(sub.columns, pd.MultiIndex):
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

def calculate_indicators(ticker, start_date, end_date):
    """
    Calculate technical indicators for a given ticker.
    Uses preloaded PRICE_DATA if available (speeds up dramatically).
    """
    try:
        # Prefer cached batch data if present
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

    if all_methods_signals > 0:
        all_signal_dates = recent_df[recent_df['All_Methods_Buy']].index
        last_all_signal = all_signal_dates.max() if len(all_signal_dates) > 0 else None

        # df now always has single-level "Close" because cache flattens per ticker
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_ema5 = float(df['EMA5'].iloc[-1])
        current_ema20 = float(df['EMA20'].iloc[-1])
        current_sma20 = float(df['SMA20'].iloc[-1])
        current_sma50 = float(df['SMA50'].iloc[-1])

        ema_strength = ((current_ema5 - current_ema20) / current_ema20) * 100 if current_ema20 else 0.0
        sma_strength = ((current_sma20 - current_sma50) / current_sma50) * 100 if current_sma50 else 0.0
        bb_position = ((current_price - df['BB_Lower'].iloc[-1]) / df['BB_Range'].iloc[-1]) * 100

        # Avoid slow .info; fast_info is OK for mcap
                # Avoid slow .info; fast_info is OK for mcap
        try:
            ti = yf.Ticker(ticker)
            fi = getattr(ti, "fast_info", None)
            
            # Try multiple ways to get company name
            company_name = "N/A"
            if fi:
                # Try different company name fields
                company_name = getattr(fi, "long_name", None)
                if not company_name:
                    company_name = getattr(fi, "short_name", None)
                if not company_name:
                    company_name = getattr(fi, "display_name", None)
                if not company_name:
                    company_name = getattr(fi, "name", None)
            
            # If still no company name, try the regular info method
            if company_name == "N/A" or not company_name:
                try:
                    info = ti.info
                    company_name = info.get('longName', info.get('shortName', info.get('displayName', ticker)))
                except:
                    company_name = ticker  # Use ticker as fallback
            
            market_cap = getattr(fi, "market_cap", np.nan) if fi else np.nan
            
            # Try to get sector information
            sector = "N/A"
            if fi:
                sector = getattr(fi, "sector", "N/A")
            if sector == "N/A":
                try:
                    info = ti.info
                    sector = info.get('sector', info.get('industry', 'N/A'))
                except:
                    sector = "N/A"
                    
        except Exception:
            company_name = ticker  # Use ticker as fallback instead of "N/A"
            market_cap = np.nan
            sector = "N/A"

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
            'Market_Cap': market_cap
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

    # Case A: level 0 = ticker, level 1 = field
    if tkr in lv0:
        sub = data[tkr]
    # Case B: level 1 = ticker, level 0 = field
    elif tkr in lv1:
        sub = data.xs(tkr, axis=1, level=1)
    else:
        return None

    # If still multiindex (rare), flatten to fields only
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(0)

    # Keep only standard OHLCV columns that exist
    cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in sub.columns]
    if not cols:
        return None
    sub = sub[cols].dropna(how='all')
    return sub if not sub.empty else None

# =========================
# Main screening (chunked)
# =========================
def screen_nasdaq_multi_methodology(
    min_avg_volume_20=MIN_AVG_VOLUME_20,
    min_last_close=MIN_LAST_CLOSE,
    batch_size=BATCH_SIZE,
    lb_days=lookback_days,
    _tickers_override=None
):
    """
    Screen Nasdaq 100 stocks for buy signals from ALL methodologies.
    Uses fast chunked downloads and a global PRICE_DATA cache.
    """
    tickers = _tickers_override or get_nasdaq_tickers()
    n_tickers = len(tickers)
    multi_buy_signal_stocks = []

    # Reset cache for a clean run
    PRICE_DATA.clear()

    print(f"\nPrefetching historical data in batches of {batch_size}...")
    for i, batch in enumerate(_chunks(tickers, batch_size), start=1):
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False,
                               group_by='ticker', threads=True)
            # Populate cache (FIXED: handle both orientations)
            if isinstance(data.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        sub = _slice_from_batch(data, tkr)
                        if sub is not None and not sub.empty:
                            PRICE_DATA[tkr] = sub
                    except Exception:
                        pass
            else:
                # Rare case: single ticker or provider quirk
                for tkr in batch:
                    if not data.empty:
                        PRICE_DATA[tkr] = data.dropna().copy()
        except Exception as e:
            print(f"Batch {i}: error during download -> {e}")

        if i % 5 == 0:
            print(f"  Prefetched ~{min(i*batch_size, n_tickers)}/{n_tickers} symbols")

    print(f"Prefetch complete. Cached {len(PRICE_DATA)} tickers.\n")

    # Liquidity filter
    if min_avg_volume_20 > 0 or min_last_close > 0.0:
        filtered_universe = []
        def _passes_liquidity(tkr):
            df = PRICE_DATA.get(tkr)
            if df is None or df.empty or 'Volume' not in df.columns or 'Close' not in df.columns or len(df) < 20:
                # Tiny fallback if batch slice failed for this ticker only
                tmp = yf.download(tkr, period="3mo", interval="1d", progress=False, threads=False)
                if tmp is None or tmp.empty or 'Volume' not in tmp.columns or 'Close' not in tmp.columns or len(tmp) < 20:
                    return False
                df = tmp

            vol_ok = float(df['Volume'].tail(20).mean()) >= float(min_avg_volume_20) if min_avg_volume_20 > 0 else True
            px_ok = float(df['Close'].iloc[-1]) >= float(min_last_close) if min_last_close > 0 else True
            return bool(vol_ok and px_ok)

        for tkr in tickers:
            try:
                if _passes_liquidity(tkr):
                    filtered_universe.append(tkr)
            except Exception:
                pass
        print(f"Liquidity filter: {len(filtered_universe)}/{n_tickers} tickers")
    else:
        filtered_universe = tickers

    print(f"\nScreening {len(filtered_universe)} Nasdaq stocks with multi-methodology approach...")
    print("This may take several minutes...")

    for i, ticker in enumerate(filtered_universe, start=1):
        try:
            if (i % 100) == 0:
                print(f"Processed {i}/{len(filtered_universe)} stocks...")

            df = calculate_indicators(ticker, start_date, end_date)
            result = check_recent_multi_buy_signals(df, ticker, lb_days)

            if result:
                multi_buy_signal_stocks.append(result)
                print(f"✅ {ticker}: Multi-methodology buy signal found!")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue

    return multi_buy_signal_stocks, len(filtered_universe)

# =========================
# Pretty console output
# =========================
def _print_console(results, total_screened):
    print("\n" + "=" * 80)
    print("🎯 NASDAQ 100 STOCKS WITH BUY SIGNALS FROM ALL METHODOLOGIES")
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
                        'All_Methods_Signals', 'Quality_Signals', 'Last_Signal_Date']
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

        print(f"\n🏆 TOP 5 STRONGEST MULTI-METHODOLOGY SIGNALS:")
        top_5 = results_df.head(5)
        for _, row in top_5.iterrows():
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

    results, total_screened = screen_nasdaq_multi_methodology(
        min_avg_volume_20=MIN_AVG_VOLUME_20,
        min_last_close=MIN_LAST_CLOSE,
        batch_size=BATCH_SIZE,
        lb_days=lookback_days
    )
    _print_console(results, total_screened)

# =========================
# Streamlit app (optional)
# =========================
def run_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    from datetime import timedelta

    st.set_page_config(page_title="Nasdaq 100 Multi-Method Screener", layout="wide")
    st.title("📈 Nasdaq 100 Multi-Methodology Screener")
    st.caption("SMA crossover • EMA+RSI • Bollinger bounce/cross • Liquidity filter")

    with st.sidebar:
        st.subheader("Scan settings")
        min_vol = st.number_input("Min 20-day avg volume", value=int(MIN_AVG_VOLUME_20), step=50_000, min_value=0)
        min_px = st.number_input("Min last close ($)", value=float(MIN_LAST_CLOSE), step=0.5, min_value=0.0, format="%.2f")
        lb_days = st.number_input("Lookback bars", value=int(lookback_days), step=1, min_value=3, max_value=30)
        batch = st.number_input("Batch size", value=int(BATCH_SIZE), step=20, min_value=20, max_value=200)

        run_btn = st.button("Run scan now")
        st.caption(f"Window: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

    @st.cache_data(ttl=24*3600, show_spinner=False)
    def run_scan_cached(min_vol, min_px, batch, lb_days):
        PRICE_DATA.clear()
        results, total = screen_nasdaq_multi_methodology(
            min_avg_volume_20=min_vol,
            min_last_close=min_px,
            batch_size=batch,
            lb_days=lb_days
        )
        df = pd.DataFrame(results) if results else pd.DataFrame()
        if not df.empty:
            df["Combined_Strength"] = df["EMA_Strength_%"] + df["SMA_Strength_%"]
            df = df.sort_values("Combined_Strength", ascending=False)
        return df, total

    if run_btn:
        st.toast("Running fresh scan… this can take several minutes.", icon="⏳")
        st.cache_data.clear()

    results_df, total_screened = run_scan_cached(min_vol, min_px, batch, lb_days)

    if results_df.empty:
        st.info("No stocks found with buy signals from ALL methodologies in the last window.")
        return

    quality_signals = int(results_df["Quality_Signals"].sum()) if "Quality_Signals" in results_df else 0
    success_rate = 100.0 * len(results_df) / max(total_screened, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total screened", f"{total_screened:,}")
    m2.metric("Multi-method signals", f"{len(results_df):,}")
    m3.metric("Quality signals (vol)", f"{quality_signals:,}")
    m4.metric("Hit rate", f"{success_rate:.2f}%")

    # Top 5 by Combined Strength - MOVED TO TOP
    st.subheader("🏆 Top 5 by Combined Strength")
    top5 = results_df.head(5)
    if not top5.empty:
        st.table(top5[['Ticker', 'Company', 'EMA_Strength_%', 'SMA_Strength_%', 'RSI', 'BB_Position_%']])
    
    # Multi-Methodology Plotting Section - MOVED ABOVE METHODOLOGY BREAKDOWN
    st.subheader("📈 Multi-Methodology Signal Plots")
    
    # Plot selector
    plot_ticker = st.selectbox(
        "Select ticker to plot:",
        options=results_df['Ticker'].tolist(),
        index=0
    )
    
    if plot_ticker:
        try:
            # Get company name
            company = results_df[results_df['Ticker'] == plot_ticker]['Company'].iloc[0]
            
            # Download data for plotting
            plot_start = end_date - timedelta(days=180)
            df = yf.download(plot_ticker, start=plot_start, end=end_date, progress=False)
            
            if not df.empty:
                # Handle multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    close_col = df[("Close", plot_ticker)] if ("Close", plot_ticker) in df.columns else df["Close"]
                    high_col = df[("High", plot_ticker)] if ("High", plot_ticker) in df.columns else df["High"]
                    low_col = df[("Low", plot_ticker)] if ("Low", plot_ticker) in df.columns else df["Low"]
                    volume_col = df[("Volume", plot_ticker)] if ("Volume", plot_ticker) in df.columns else df["Volume"]
                else:
                    close_col = df["Close"]
                    high_col = df["High"]
                    low_col = df["Low"]
                    volume_col = df["Volume"]

                # Calculate indicators (same as screening logic)
                # SMA crossover
                df["SMA20"] = close_col.rolling(20).mean()
                df["SMA50"] = close_col.rolling(50).mean()
                df["SMA_Buy_Signal"] = (df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
                df["SMA_Sell_Signal"] = (df["SMA20"] < df["SMA50"]) & (df["SMA20"].shift(1) >= df["SMA50"].shift(1))

                # EMA & RSI
                df["EMA5"] = close_col.ewm(span=5, adjust=False).mean()
                df["EMA20"] = close_col.ewm(span=20, adjust=False).mean()
                delta = close_col.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))
                df["EMA_RSI_Buy_Signal"] = (df["EMA5"] > df["EMA20"]) & (df["RSI"] > 50) & (df["RSI"] < 70)
                df["EMA_RSI_Sell_Signal"] = (df["EMA5"] < df["EMA20"]) & (df["RSI"] < 50) & (df["RSI"] > 30)

                # Bollinger Bands
                df["BB_Middle"] = close_col.rolling(20).mean()
                bb_std = close_col.rolling(20).std()
                df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
                df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
                df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
                bb_lower_bounce = (close_col.shift(1) <= df['BB_Lower'].shift(1) * 1.02) & (close_col > df['BB_Lower'] * 1.02)
                bb_middle_cross = (close_col > df['BB_Middle']) & (close_col.shift(1) <= df['BB_Middle'].shift(1))
                bb_width_ok = df['BB_Width'] > df['BB_Width'].rolling(20).mean() * 0.8
                df["BB_Buy_Signal"] = (bb_lower_bounce | bb_middle_cross) & bb_width_ok
                bb_upper_bounce = (close_col.shift(1) >= df['BB_Upper'].shift(1) * 0.98) & (close_col < df['BB_Upper'] * 0.98)
                bb_middle_cross_down = (close_col < df['BB_Middle']) & (close_col.shift(1) >= df['BB_Middle'].shift(1))
                df["BB_Sell_Signal"] = (bb_upper_bounce | bb_middle_cross_down) & bb_width_ok

                # Combined multi-method signals
                df["All_Methods_Buy"] = ((df["SMA_Buy_Signal"] | (df["SMA20"] > df["SMA50"])) &
                                          df["EMA_RSI_Buy_Signal"] & df["BB_Buy_Signal"])
                df["All_Methods_Sell"] = ((df["SMA_Sell_Signal"] | (df["SMA20"] < df["SMA50"])) &
                                           df["EMA_RSI_Sell_Signal"] & df["BB_Sell_Signal"])

                df = df.dropna()
                
                if not df.empty:
                    # Create the plot
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

                    # Price chart with indicators
                    ax1.plot(df.index, df["Close"], label="Close", alpha=0.85, linewidth=1)
                    ax1.plot(df.index, df["SMA20"], label="SMA20", linewidth=1.2, color='steelblue')
                    ax1.plot(df.index, df["SMA50"], label="SMA50", linewidth=1.2, color='orange')
                    ax1.plot(df.index, df["EMA5"], label="EMA5", linewidth=1.2, color='green', alpha=0.8)
                    ax1.plot(df.index, df["EMA20"], label="EMA20", linewidth=1.2, color='red', alpha=0.8)

                    # Multi-method signals
                    multi_buy = df[df["All_Methods_Buy"]]
                    multi_sell = df[df["All_Methods_Sell"]]
                    if not multi_buy.empty:
                        ax1.scatter(multi_buy.index, multi_buy["Close"], marker="^", color="darkgreen", s=140,
                                    label="Multi-Method BUY", zorder=5)
                    if not multi_sell.empty:
                        ax1.scatter(multi_sell.index, multi_sell["Close"], marker="v", color="darkred", s=140,
                                    label="Multi-Method SELL", zorder=5)

                    ax1.set_title(f"{plot_ticker} {company} - Multi-Method Buy/Sell Signals", fontsize=14, fontweight='bold')
                    ax1.set_ylabel("Price (USD)")
                    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    ax1.grid(True, alpha=0.3)

                    # RSI chart
                    ax2.plot(df.index, df["RSI"], label="RSI", color='purple', linewidth=1.5)
                    ax2.axhline(70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                    ax2.axhline(30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                    ax2.axhline(50, color='black', linestyle='-', alpha=0.5, label='Neutral (50)')
                    ax2.set_ylabel("RSI")
                    ax2.set_xlabel("Date")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)
                    
                    # Show signal counts
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Multi-Method BUY Signals", len(multi_buy))
                    with col2:
                        st.metric("Multi-Method SELL Signals", len(multi_sell))
                    with col3:
                        st.metric("Total Trading Days", len(df))
                        
                else:
                    st.warning(f"No data available for {plot_ticker} after calculating indicators.")
            else:
                st.error(f"Could not download data for {plot_ticker}")
                
        except Exception as e:
            st.error(f"Error plotting {plot_ticker}: {str(e)}")
    
    # Methodology Breakdown - MOVED BELOW PLOTTING SECTION
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
    
    # All Results Table
    st.subheader("📋 Complete Results")
    display_cols = ['Ticker', 'Company', 'Sector', 'Current_Price', 'RSI',
                    'EMA_Strength_%', 'SMA_Strength_%', 'BB_Position_%',
                    'All_Methods_Signals', 'Quality_Signals', 'Last_Signal_Date']
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
        file_name="Nasdaq_100_multi_methodology_buy_signals.csv",
        mime="text/csv",
        use_container_width=True,
    )
# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nasdaq 100 multi-method screener (CLI or Streamlit).")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (default is Streamlit).")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME_20, help="Min 20-day avg volume.")
    parser.add_argument("--min-price", type=float, default=MIN_LAST_CLOSE, help="Min last close price.")
    parser.add_argument("--lookback", type=int, default=lookback_days, help="Lookback bars for signals.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size for yfinance download.")
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        # Launch Streamlit app when executed normally:
        #   streamlit run nasdaq_multi_method_screener.py
        run_streamlit()
