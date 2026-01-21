# file: Daily_Drop_Detector_Streamlit.py

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Import universe ticker functions from myutils
from myutils import get_sp500_tickers

# =========================
# Global config
# =========================
BATCH_SIZE = 20
MIN_AVG_VOLUME_20 = 150_000  # 0 to disable
MIN_LAST_CLOSE = 3.0         # 0 to disable
DROP_THRESHOLD_PCT = 10.0    # 10% drop threshold

# Default time window (past year for daily data)
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# =========================
# Helper: chunking
# =========================
def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# Get company info
# =========================
def get_company_info_robust(ticker, max_retries=2):
    """Get company name and sector."""
    import time
    for attempt in range(max_retries):
        try:
            ti = yf.Ticker(ticker)
            info = {}
            try:
                if hasattr(ti, "get_info"):
                    info = ti.get_info() or {}
                else:
                    info = ti.info or {}
            except Exception:
                info = {}
            
            company_name = ticker
            sector = "N/A"
            
            name_candidate = info.get('longName') or info.get('shortName') or info.get('displayName')
            if name_candidate and str(name_candidate).strip():
                company_name = str(name_candidate).strip()
            
            sec_candidate = info.get('sector') or info.get('industry')
            if sec_candidate and str(sec_candidate).strip():
                sector = str(sec_candidate).strip()
            
            return company_name, sector
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.3)
                continue
    return ticker, "N/A"

# =========================
# Detect daily drops
# =========================
def detect_daily_drops(tickers, drop_threshold_pct, min_volume=0, min_price=0, batch_size=20):
    """
    Scan tickers for daily drops >= drop_threshold_pct.
    Returns list of dictionaries with drop information.
    """
    results = []
    price_data = {}
    
    print(f"📥 Downloading daily data for {len(tickers)} tickers...")
    
    # Download data in batches
    for i, batch in enumerate(_chunks(tickers, batch_size), start=1):
        try:
            print(f"  Batch {i}/{len(list(_chunks(tickers, batch_size)))}: {len(batch)} tickers...")
            
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=True
            )
            
            if data is None or data.empty:
                continue
            
            # Extract data for each ticker
            if isinstance(data.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        if tkr in data.columns.get_level_values(0):
                            sub = data[tkr]
                        elif tkr in data.columns.get_level_values(1):
                            sub = data.xs(tkr, axis=1, level=1)
                        else:
                            continue
                        
                        if isinstance(sub.columns, pd.MultiIndex):
                            sub.columns = sub.columns.get_level_values(0)
                        
                        if 'Close' in sub.columns and not sub.empty:
                            price_data[tkr] = sub
                    except Exception:
                        continue
            else:
                # Single ticker case
                if len(batch) == 1 and 'Close' in data.columns:
                    price_data[batch[0]] = data
        except Exception as e:
            print(f"    ⚠️  Batch error: {e}")
            continue
    
    print(f"✅ Downloaded data for {len(price_data)} tickers")
    print(f"🔍 Scanning for drops >= {drop_threshold_pct}%...")
    
    # Scan for drops
    for ticker, df in price_data.items():
        try:
            if df.empty or len(df) < 2:
                continue
            
            # Apply liquidity filters
            if min_volume > 0:
                volume_col = df['Volume'] if 'Volume' in df.columns else pd.Series()
                if not volume_col.empty and len(volume_col) >= 20:
                    avg_volume = volume_col.tail(20).mean()
                    if avg_volume < min_volume:
                        continue
            
            if min_price > 0:
                current_price = float(df['Close'].iloc[-1])
                if current_price < min_price:
                    continue
            
            # Calculate daily percentage change
            close_prices = df['Close'].astype(float)
            daily_pct_change = close_prices.pct_change() * 100.0
            
            # Find drops >= threshold
            drops = daily_pct_change[daily_pct_change <= -drop_threshold_pct]
            
            if not drops.empty:
                # Get most recent drop
                latest_drop_date = drops.index[-1]
                latest_drop_pct = float(drops.iloc[-1])
                
                # Get prices
                drop_day_price = float(close_prices.loc[latest_drop_date])
                prev_day_price = float(close_prices.shift(1).loc[latest_drop_date])
                
                # Calculate metrics
                current_price = float(close_prices.iloc[-1])
                volume_on_drop = float(df.loc[latest_drop_date, 'Volume']) if 'Volume' in df.columns else 0
                
                # 52-week high/low context
                lookback_252 = min(252, len(close_prices))
                high_52w = float(close_prices.tail(lookback_252).max())
                low_52w = float(close_prices.tail(lookback_252).min())
                price_from_52w_high = ((current_price - high_52w) / high_52w) * 100.0
                
                # Count total drops in period
                total_drops = len(drops)
                
                # Get company info
                company_name, sector = get_company_info_robust(ticker, max_retries=1)
                
                results.append({
                    'Ticker': ticker,
                    'Company': company_name,
                    'Sector': sector,
                    'Drop_Date': latest_drop_date.strftime('%Y-%m-%d') if hasattr(latest_drop_date, 'strftime') else str(latest_drop_date),
                    'Drop_Pct': round(latest_drop_pct, 2),
                    'Price_On_Drop': round(drop_day_price, 2),
                    'Prev_Day_Price': round(prev_day_price, 2),
                    'Current_Price': round(current_price, 2),
                    'Volume_On_Drop': int(volume_on_drop) if volume_on_drop > 0 else 0,
                    'Price_From_52W_High': round(price_from_52w_high, 2),
                    '52W_High': round(high_52w, 2),
                    '52W_Low': round(low_52w, 2),
                    'Total_Drops_Period': total_drops,
                    'Days_Since_Drop': (df.index[-1] - latest_drop_date).days if hasattr(df.index[-1], '__sub__') else 0
                })
                
        except Exception as e:
            print(f"  ⚠️  Error processing {ticker}: {str(e)[:50]}")
            continue
    
    # Sort by most recent drop and severity
    results.sort(key=lambda x: (x['Days_Since_Drop'], x['Drop_Pct']))
    
    return results

# =========================
# Streamlit app
# =========================
def run_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    
    st.set_page_config(page_title="Daily Drop Detector", layout="wide")
    st.title("📉 Daily Stock Drop Detector")
    st.caption(f"Flags stocks with drops >= 10% on a trading day basis")
    
    with st.sidebar:
        st.subheader("Scan Settings")
        universe = st.selectbox("Universe", options=["S&P 500"], index=0)
        drop_threshold = st.number_input("Drop Threshold (%)", value=float(DROP_THRESHOLD_PCT), 
                                         min_value=1.0, max_value=50.0, step=0.5)
        min_vol = st.number_input("Min 20-day avg volume", value=int(MIN_AVG_VOLUME_20), 
                                  step=50_000, min_value=0)
        min_px = st.number_input("Min last close ($)", value=float(MIN_LAST_CLOSE), 
                                 step=0.5, min_value=0.0, format="%.2f")
        batch = st.number_input("Batch size", value=int(BATCH_SIZE), step=5, min_value=5, max_value=50)
        
        run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)
        st.caption(f"Window: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_scan_cached(universe_name, threshold, min_vol, min_px, batch_size):
        if universe_name == "S&P 500":
            tickers = get_sp500_tickers()
        results = detect_daily_drops(
            tickers=tickers,
            drop_threshold_pct=threshold,
            min_volume=min_vol,
            min_price=min_px,
            batch_size=batch_size
        )
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    if run_btn:
        st.cache_data.clear()
        st.toast("Running scan... this may take a few minutes.", icon="⏳")
    
    results_df = run_scan_cached(universe, drop_threshold, min_vol, min_px, batch)
    
    if results_df.empty:
        st.info("No stocks found with drops >= {:.1f}% in the scan period.".format(drop_threshold))
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks with Drops", len(results_df))
        with col2:
            avg_drop = results_df['Drop_Pct'].mean()
            st.metric("Average Drop %", f"{avg_drop:.2f}%")
        with col3:
            max_drop = results_df['Drop_Pct'].min()  # Most negative
            st.metric("Largest Drop %", f"{max_drop:.2f}%")
        with col4:
            recent_drops = len(results_df[results_df['Days_Since_Drop'] <= 5])
            st.metric("Drops (Last 5 Days)", recent_drops)
        
        # Filters
        st.subheader("📊 Filter Results")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            min_drop = st.number_input("Min Drop %", value=float(drop_threshold), 
                                       min_value=1.0, max_value=50.0, step=0.5)
        with filter_col2:
            max_days = st.number_input("Max Days Since Drop", value=365, 
                                       min_value=1, max_value=365)
        with filter_col3:
            sector_filter = st.selectbox("Filter by Sector", 
                                        options=["All"] + sorted(results_df['Sector'].unique().tolist()))
        
        # Apply filters
        filtered_df = results_df[
            (results_df['Drop_Pct'] <= -min_drop) & 
            (results_df['Days_Since_Drop'] <= max_days)
        ].copy()
        
        if sector_filter != "All":
            filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
        
        if filtered_df.empty:
            st.warning("No results match the selected filters.")
        else:
            # Display table
            st.subheader(f"📋 Results ({len(filtered_df)} stocks)")
            
            display_cols = ['Ticker', 'Company', 'Sector', 'Drop_Date', 'Drop_Pct', 
                          'Price_On_Drop', 'Current_Price', 'Price_From_52W_High', 
                          'Days_Since_Drop', 'Total_Drops_Period']
            
            show_df = filtered_df[display_cols].copy()
            show_df = show_df.sort_values(['Days_Since_Drop', 'Drop_Pct'])
            
            st.dataframe(show_df, use_container_width=True, height=400)
            
            # Download button
            csv = show_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"daily_drops_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.subheader("📈 Visualizations")
            
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.write("**Drops by Sector**")
                sector_counts = filtered_df.groupby('Sector').size().sort_values(ascending=False)
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sector_counts.head(10).plot(kind='barh', ax=ax1)
                ax1.set_xlabel('Number of Drops')
                ax1.set_title('Top 10 Sectors with Most Drops')
                plt.tight_layout()
                st.pyplot(fig1)
            
            with vis_col2:
                st.write("**Drop Distribution**")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                filtered_df['Drop_Pct'].hist(bins=30, ax=ax2, edgecolor='black')
                ax2.axvline(filtered_df['Drop_Pct'].mean(), color='r', linestyle='--', 
                           label=f"Mean: {filtered_df['Drop_Pct'].mean():.2f}%")
                ax2.set_xlabel('Drop Percentage (%)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Drop Percentages')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Individual stock plot
            st.write("**Individual Stock Analysis**")
            plot_ticker = st.selectbox(
                "Select a ticker to plot",
                options=sorted(filtered_df['Ticker'].unique()),
                index=0
            )
            
            if plot_ticker:
                try:
                    # Download data for plotting
                    ticker_data = yf.download(plot_ticker, start=start_date, end=end_date, progress=False)
                    
                    if not ticker_data.empty and 'Close' in ticker_data.columns:
                        # Get drop dates for this ticker
                        ticker_drops = filtered_df[filtered_df['Ticker'] == plot_ticker]
                        
                        fig3, ax3 = plt.subplots(figsize=(12, 6))
                        
                        # Plot price
                        ax3.plot(ticker_data.index, ticker_data['Close'], 
                                label='Close Price', linewidth=1.5, color='blue')
                        
                        # Mark drop dates
                        for _, drop in ticker_drops.iterrows():
                            drop_date = pd.to_datetime(drop['Drop_Date'])
                            if drop_date in ticker_data.index:
                                drop_price = ticker_data.loc[drop_date, 'Close']
                                ax3.scatter(drop_date, drop_price, 
                                          color='red', s=200, marker='v', 
                                          zorder=5, label='Drop Event' if drop == ticker_drops.iloc[0] else "")
                        
                        # Formatting
                        company_name = ticker_drops.iloc[0]['Company']
                        ax3.set_title(f"{plot_ticker} - {company_name} (Drops >= {drop_threshold}%)", 
                                    fontsize=14, fontweight='bold')
                        ax3.set_xlabel('Date')
                        ax3.set_ylabel('Price (USD)')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        
                        # Show drop details
                        st.dataframe(ticker_drops[['Drop_Date', 'Drop_Pct', 'Price_On_Drop', 
                                                  'Prev_Day_Price', 'Volume_On_Drop']], 
                                   use_container_width=True)
                    else:
                        st.error(f"Could not download data for {plot_ticker}")
                        
                except Exception as e:
                    st.error(f"Error plotting {plot_ticker}: {str(e)}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Check if streamlit is available - if so, run directly
    try:
        import streamlit
        run_streamlit()
    except ImportError:
        # Streamlit not installed or not being used
        parser = argparse.ArgumentParser(description="Daily stock drop detector with Streamlit visualization.")
        parser.add_argument("--cli", action="store_true", help="Run in CLI mode (not implemented - use Streamlit)")
        args, unknown = parser.parse_known_args()
        
        if args.cli:
            print("⚠️  CLI mode not implemented.")
        print("📊 Use Streamlit to run this app:")
        print("   streamlit run Daily_Drop_Detector_Streamlit.py")