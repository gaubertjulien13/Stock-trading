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

#get nasdaq composite tickers

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

#get sp500 tickers

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