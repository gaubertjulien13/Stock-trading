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
# Centralized Ticker Management with Weekly Caching
# =========================

def get_sp500_tickers(use_cache=True, cache_path="sp500_tickers_cache.json.gz", max_cache_age_hours=168):  # 168 hours = 1 week
    """
    Return a list of S&P 500 ticker symbols (normalized for yfinance).
    Strategy: try multiple data sources with graceful fallback.
    - Primary: GitHub dataset (CSV)
    - Secondary: StockAnalysis.com list (HTML table)
    - Tertiary: Wikipedia (HTML table, with headers)
    Caches results locally for 1 week to avoid frequent API calls.
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

    # ---------- Check weekly cache ----------
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
                cache_age_days = age_hours / 24.0
                print(f"📋 Using cached S&P 500 tickers ({len(payload['tickers'])} tickers, {cache_age_days:.1f} days old)")
                return payload["tickers"]
            else:
                print(f"🔄 S&P 500 cache expired ({age_hours/24:.1f} days old), refreshing...")
        except Exception:
            print("⚠️ S&P 500 cache corrupted, refreshing...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    })
    timeout = 20

    errors = []

    # ---------- Source 1: GitHub CSV (datasets/s-and-p-500-companies) ----------
    try:
        print("📥 Fetching fresh S&P 500 tickers from GitHub...")
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "Symbol" in df.columns:
            tickers = _normalize(df["Symbol"])
            if _valid(tickers):
                # Save to weekly cache
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers, "source": "GitHub CSV"}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                    print(f"✅ S&P 500 tickers cached for 1 week ({len(tickers)} tickers)")
                except Exception:
                    pass
                return tickers
        else:
            errors.append("GitHub CSV missing 'Symbol' column")
    except Exception as e:
        errors.append(f"GitHub CSV failed: {e}")

    # ---------- Source 2: StockAnalysis.com table ----------
    try:
        print("📥 Trying StockAnalysis.com as fallback...")
        url = "https://stockanalysis.com/list/sp-500-stocks/"
        tables = pd.read_html(session.get(url, timeout=timeout).text)
        df = next((t for t in tables if any(c.lower() == "symbol" for c in map(str, t.columns))), None)
        if df is not None:
            symbol_col = next(c for c in df.columns if str(c).lower() == "symbol")
            tickers = _normalize(df[symbol_col])
            if _valid(tickers):
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers, "source": "StockAnalysis"}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                    print(f"✅ S&P 500 tickers cached for 1 week ({len(tickers)} tickers)")
                except Exception:
                    pass
                return tickers
        else:
            errors.append("StockAnalysis page: no table with Symbol found")
    except Exception as e:
        errors.append(f"StockAnalysis failed: {e}")

    # ---------- Source 3: Wikipedia ----------
    try:
        print("📥 Trying Wikipedia as final fallback...")
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
                try:
                    payload = {"timestamp": time.time(), "tickers": tickers, "source": "Wikipedia"}
                    if cache_path.endswith(".gz"):
                        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f)
                    else:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                    print(f"✅ S&P 500 tickers cached for 1 week ({len(tickers)} tickers)")
                except Exception:
                    pass
                return tickers
        else:
            errors.append("Wikipedia: constituents table not found")
    except Exception as e:
        errors.append(f"Wikipedia failed: {e}")

    # ---------- Final fallback ----------
    print("⚠️ All sources failed. Using hardcoded fallback tickers. Errors:", " | ".join(errors))
    return ["AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "NVDA", "BRK-B", "UNH", "XOM"]


def get_nasdaq_composite_tickers(use_cache=True, cache_path="nasdaq_tickers_cache.json.gz", max_cache_age_hours=168):  # 168 hours = 1 week
    """
    Fetch Nasdaq-listed common stocks from Nasdaq Trader (HTTPS),
    filtering out ETFs, test issues, NextShares, etc.
    Caches results locally for 1 week.
    """
    # ---------- Check weekly cache ----------
    if use_cache and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".gz"):
                with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            age_hours = (time.time() - payload["timestamp"]) / 3600.0
            if age_hours <= max_cache_age_hours and len(payload.get("tickers", [])) > 1000:  # Expect 3000+ Nasdaq tickers
                cache_age_days = age_hours / 24.0
                print(f"📋 Using cached Nasdaq tickers ({len(payload['tickers'])} tickers, {cache_age_days:.1f} days old)")
                return payload["tickers"]
            else:
                print(f"🔄 Nasdaq cache expired ({age_hours/24:.1f} days old), refreshing...")
        except Exception:
            print("⚠️ Nasdaq cache corrupted, refreshing...")

    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        print("📥 Fetching fresh Nasdaq tickers...")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep="|")
        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["NextShares"] == "N")]
        tickers = df["Symbol"].dropna().tolist()
        
        # Save to weekly cache
        try:
            payload = {"timestamp": time.time(), "tickers": tickers, "source": "Nasdaq Trader"}
            if cache_path.endswith(".gz"):
                with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                    json.dump(payload, f)
            else:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            print(f"✅ Nasdaq tickers cached for 1 week ({len(tickers)} tickers)")
        except Exception:
            pass
            
        return tickers
    except Exception as e:
        print(f"⚠️ Error fetching Nasdaq list: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']


# =========================
# Ticker Management Utilities
# =========================

def refresh_all_ticker_caches():
    """
    Force refresh of both S&P 500 and Nasdaq ticker caches.
    Useful for weekly maintenance or when you suspect stale data.
    """
    print("🔄 Force refreshing all ticker caches...")
    
    # Remove existing cache files
    for cache_file in ["sp500_tickers_cache.json.gz", "nasdaq_tickers_cache.json.gz"]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"🗑️ Removed old cache: {cache_file}")
    
    # Fetch fresh data
    sp500_tickers = get_sp500_tickers(use_cache=False)
    nasdaq_tickers = get_nasdaq_composite_tickers(use_cache=False)
    
    print(f"✅ Refreshed S&P 500: {len(sp500_tickers)} tickers")
    print(f"✅ Refreshed Nasdaq: {len(nasdaq_tickers)} tickers")
    
    return sp500_tickers, nasdaq_tickers


def get_cache_status():
    """
    Check the age and status of ticker caches.
    """
    caches = [
        ("S&P 500", "sp500_tickers_cache.json.gz"),
        ("Nasdaq", "nasdaq_tickers_cache.json.gz")
    ]
    
    print("📊 Ticker Cache Status:")
    print("-" * 50)
    
    for name, cache_path in caches:
        if os.path.exists(cache_path):
            try:
                if cache_path.endswith(".gz"):
                    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                        payload = json.load(f)
                else:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                
                age_hours = (time.time() - payload["timestamp"]) / 3600.0
                age_days = age_hours / 24.0
                ticker_count = len(payload.get("tickers", []))
                source = payload.get("source", "Unknown")
                
                status = "✅ Fresh" if age_hours <= 168 else "⚠️ Expired"
                print(f"{name:10} | {status} | {age_days:5.1f} days old | {ticker_count:4d} tickers | {source}")
                
            except Exception as e:
                print(f"{name:10} | ❌ Corrupted | Error: {str(e)[:30]}...")
        else:
            print(f"{name:10} | 🆕 No cache | Will fetch on next use")


# =========================
# Convenience Functions
# =========================

def get_universe_tickers(universe="sp500", use_cache=True):
    """
    Convenience function to get tickers for a specific universe.
    
    Args:
        universe (str): "sp500", "nasdaq", or "both"
        use_cache (bool): Whether to use cached tickers
    
    Returns:
        list: Ticker symbols
    """
    if universe.lower() in ["sp500", "sp", "s&p500"]:
        return get_sp500_tickers(use_cache=use_cache)
    elif universe.lower() in ["nasdaq", "nq", "composite"]:
        return get_nasdaq_composite_tickers(use_cache=use_cache)
    elif universe.lower() == "both":
        sp500 = get_sp500_tickers(use_cache=use_cache)
        nasdaq = get_nasdaq_composite_tickers(use_cache=use_cache)
        # Combine and deduplicate
        combined = list(set(sp500 + nasdaq))
        print(f"📊 Combined universe: {len(combined)} unique tickers (S&P500: {len(sp500)}, Nasdaq: {len(nasdaq)})")
        return combined
    else:
        raise ValueError(f"Unknown universe: {universe}. Use 'sp500', 'nasdaq', or 'both'")


if __name__ == "__main__":
    """
    Command-line interface for ticker management.
    Usage:
        python myutils.py status        # Check cache status
        python myutils.py refresh       # Force refresh all caches
        python myutils.py sp500         # Get S&P 500 tickers
        python myutils.py nasdaq        # Get Nasdaq tickers
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python myutils.py [status|refresh|sp500|nasdaq]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        get_cache_status()
    elif command == "refresh":
        refresh_all_ticker_caches()
    elif command == "sp500":
        tickers = get_sp500_tickers()
        print(f"S&P 500 tickers: {tickers[:10]}... ({len(tickers)} total)")
    elif command == "nasdaq":
        tickers = get_nasdaq_composite_tickers()
        print(f"Nasdaq tickers: {tickers[:10]}... ({len(tickers)} total)")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)