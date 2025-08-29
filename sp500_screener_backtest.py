#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SP500 Screener Backtest (Loose vs Strict + Exit Styles)

Usage examples:
  python sp500_screener_backtest.py --years 5 --batch 60 --threads --max-hold 60 --atr-k 2.0
  python sp500_screener_backtest.py --start 2020-01-01 --end 2025-08-01 --exit-styles fixed5,fixed10,fixed20,atr2,atr1_5,smaCross

It will output:
  - backtest_summary.csv : Aggregate metrics per (screener, exit_style)
  - trade_log.csv        : Per-trade entries/exits and returns

Notes:
  - Requires: pandas, numpy, yfinance, requests, lxml (for Wikipedia table), matplotlib optional
  - Internet access is required to fetch prices.
"""

import argparse
import sys
import time
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# ------------------------------
# Universe
# ------------------------------
def get_sp500_tickers(timeout: int = 20) -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    sp500 = next((t for t in tables if set(["Symbol", "Security"]).issubset(t.columns)), None)
    if sp500 is None:
        raise RuntimeError("Could not locate S&P 500 table on the Wikipedia page.")
    tickers = (
        sp500["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).dropna().unique().tolist()
    )
    return tickers

# ------------------------------
# Helpers
# ------------------------------
def chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def slice_from_batch(data: pd.DataFrame, tkr: str) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None
    if not isinstance(data.columns, pd.MultiIndex):
        return data.dropna().copy()
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
    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in sub.columns]
    if not cols:
        return None
    sub = sub[cols].dropna(how="all")
    return sub if not sub.empty else None

def extract_cols(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close_col = df["Close"].astype(float)
    high_col  = df["High"].astype(float)
    low_col   = df["Low"].astype(float)
    vol_col   = df["Volume"].astype(float)
    return close_col, high_col, low_col, vol_col

# ------------------------------
# Indicators & Signals
# ------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    close, high, low, vol = extract_cols(df)
    # SMA / EMA
    df["SMA20"] = close.rolling(20, min_periods=20).mean()
    df["SMA50"] = close.rolling(50, min_periods=50).mean()
    df["SMA200"] = close.rolling(200, min_periods=200).mean()

    df["EMA5"] = close.ewm(span=5, adjust=False).mean()
    df["EMA20"]= close.ewm(span=20, adjust=False).mean()

    # RSI (simple rolling variant)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    # Bollinger
    df["BB_Middle"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]).abs()
    df["BB_Range"] = (df["BB_Upper"] - df["BB_Lower"]).abs().clip(lower=1e-9)

    # True Range & ATR (simple rolling mean; change to Wilder with ewm if desired)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14, min_periods=14).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # Volume avg
    df["Volume_MA20"] = vol.rolling(20, min_periods=20).mean()

    # Base method signals
    df["SMA_Buy_Signal"] = (df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
    bb_lower_bounce = (close.shift(1) <= df["BB_Lower"].shift(1) * 1.02) & (close > df["BB_Lower"] * 1.02)
    bb_middle_cross = (close > df["BB_Middle"]) & (close.shift(1) <= df["BB_Middle"].shift(1))
    bb_width_ok = df["BB_Width"] > df["BB_Width"].rolling(20, min_periods=20).mean() * 0.8
    df["BB_Buy_Signal"] = (bb_lower_bounce | bb_middle_cross) & bb_width_ok
    df["EMA_RSI_Buy_Signal"] = (df["EMA5"] > df["EMA20"]) & df["RSI"].between(50, 70)

    # Combined loose
    df["All_Methods_Buy_Loose"] = (
        (df["SMA_Buy_Signal"] | (df["SMA20"] > df["SMA50"])) &
        df["EMA_RSI_Buy_Signal"] &
        df["BB_Buy_Signal"]
    )
    # Strict: require trend + MACD bullish or recent cross
    macd_cross_up = (df["MACD_Line"] > df["MACD_Signal"]) & (df["MACD_Line"].shift(1) <= df["MACD_Signal"].shift(1))
    macd_bull = (df["MACD_Line"] > df["MACD_Signal"])
    trend_ok = df["Close"] > df["SMA200"]
    df["All_Methods_Buy_Strict"] = df["All_Methods_Buy_Loose"] & trend_ok & (macd_bull | macd_cross_up)

    return df

# ------------------------------
# Exit styles
# ------------------------------
def simulate_trade_path(df: pd.DataFrame, entry_idx: int, exit_style: str, atr_k: float = 2.0, max_hold: int = 60) -> Tuple[int, float, float]:
    """
    Simulate from entry index forward until exit condition.
    Returns: (bars_held, pct_return, max_drawdown_pct)
    """
    n = len(df)
    if entry_idx >= n-1:
        return (0, 0.0, 0.0)
    entry_price = float(df["Close"].iloc[entry_idx])
    max_price = entry_price
    min_price_since_entry = entry_price
    bars = 0
    stop = None

    def ret(p): return (p / entry_price) - 1.0

    for i in range(entry_idx+1, min(n, entry_idx + max_hold + 1)):
        price = float(df["Close"].iloc[i])
        max_price = max(max_price, price)
        min_price_since_entry = min(min_price_since_entry, price)
        bars = i - entry_idx

        if exit_style.startswith("fixed"):
            horizon = int(exit_style.replace("fixed", ""))
            if bars >= horizon:
                dd = (min_price_since_entry / entry_price) - 1.0
                return (bars, ret(price), dd)

        elif exit_style in ("atr2", "atr1_5"):
            k = 2.0 if exit_style == "atr2" else 1.5
            atr = float(df["ATR14"].iloc[i]) if not math.isnan(float(df["ATR14"].iloc[i])) else None
            if atr is None or atr == 0.0:
                # if ATR not available yet, skip until it is
                continue
            new_stop = price - k * atr
            if stop is None:
                stop = entry_price - k * (float(df["ATR14"].iloc[entry_idx]) if not math.isnan(float(df["ATR14"].iloc[entry_idx])) else atr)
            stop = max(stop, new_stop)
            if price < stop:
                dd = (min_price_since_entry / entry_price) - 1.0
                return (bars, ret(price), dd)

        elif exit_style == "smaCross":
            # exit when SMA20 crosses below SMA50
            sma20_now = df["SMA20"].iloc[i]
            sma50_prev = df["SMA50"].iloc[i-1]
            sma20_prev = df["SMA20"].iloc[i-1]
            sma50_now = df["SMA50"].iloc[i]
            if (sma20_now < sma50_now) and (sma20_prev >= sma50_prev):
                dd = (min_price_since_entry / entry_price) - 1.0
                return (bars, ret(price), dd)

        else:
            raise ValueError(f"Unknown exit_style: {exit_style}")

    # Fallback: max_hold reached without exit or end of data
    last_price = float(df["Close"].iloc[min(n-1, entry_idx + max_hold)])
    dd = (min_price_since_entry / entry_price) - 1.0
    return (bars, ret(last_price), dd)

# ------------------------------
# Backtest per ticker
# ------------------------------
def backtest_ticker(df: pd.DataFrame, screener: str, exit_styles: List[str], max_hold: int) -> List[Dict]:
    """
    screener: 'loose' or 'strict'
    Returns list of dicts, one per trade (with exit result per exit_style)
    """
    trades = []
    if df is None or df.empty:
        return trades

    signal_col = "All_Methods_Buy_Loose" if screener == "loose" else "All_Methods_Buy_Strict"
    # Simple rule: enter on signal True if not in position; one position at a time.
    in_pos = False
    entry_idx = None

    for i in range(len(df)):
        sig = bool(df[signal_col].iloc[i]) if signal_col in df.columns else False
        if not in_pos and sig:
            # enter at close of signal bar
            in_pos = True
            entry_idx = i
            continue

        if in_pos:
            # evaluate exits for each style independently; record separate trades for each style
            rowdate = df.index[entry_idx]
            for style in exit_styles:
                bars, pct, mdd = simulate_trade_path(df, entry_idx, style, max_hold=max_hold)
                exit_idx = entry_idx + bars
                exit_date = df.index[min(exit_idx, len(df)-1)]
                trades.append({
                    "Screener": screener,
                    "ExitStyle": style,
                    "EntryDate": rowdate,
                    "ExitDate": exit_date,
                    "BarsHeld": bars,
                    "EntryPrice": float(df["Close"].iloc[entry_idx]),
                    "ExitPrice": float(df["Close"].iloc[min(exit_idx, len(df)-1)]),
                    "ReturnPct": pct * 100.0,
                    "MaxDrawdownPct": mdd * 100.0,
                })
            # after recording, flatten position and add cooldown of 3 bars to avoid immediate re-entry
            in_pos = False
            entry_idx = None
    return trades

# ------------------------------
# Aggregation
# ------------------------------
def summarize(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    def expectancy(g):
        wins = g[g["ReturnPct"] > 0]["ReturnPct"]
        losses = g[g["ReturnPct"] <= 0]["ReturnPct"]
        win_rate = len(wins) / max(len(g), 1)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    summary = trades_df.groupby(["Screener", "ExitStyle"]).apply(
        lambda g: pd.Series({
            "Trades": len(g),
            "WinRate_%": 100.0 * (g["ReturnPct"] > 0).mean() if len(g) else 0.0,
            "AvgRet_%": g["ReturnPct"].mean() if len(g) else 0.0,
            "MedianRet_%": g["ReturnPct"].median() if len(g) else 0.0,
            "AvgBarsHeld": g["BarsHeld"].mean() if len(g) else 0.0,
            "AvgMaxDD_%": g["MaxDrawdownPct"].mean() if len(g) else 0.0,
            "Expectancy_%": expectancy(g) if len(g) else 0.0,
        })
    ).reset_index()
    # Sort by Expectancy then AvgRet
    summary = summary.sort_values(["Screener", "Expectancy_%", "AvgRet_%"], ascending=[True, False, False])
    return summary

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Backtest loose vs strict multi-method screener on S&P500.")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (overrides --years)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--years", type=int, default=5, help="Number of years back if --start not set")
    ap.add_argument("--batch", type=int, default=60, help="yfinance batch size")
    ap.add_argument("--exit-styles", type=str, default="fixed5,fixed10,fixed20,atr2,atr1_5,smaCross",
                    help="Comma list of exit styles: fixed5,fixed10,fixed20,atr2,atr1_5,smaCross")
    ap.add_argument("--max-hold", type=int, default=60, help="Max holding bars for styles that need it")
    ap.add_argument("--threads", action="store_true", help="Enable yfinance threads=True")
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.today()
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=365*args.years)

    exit_styles = [s.strip() for s in args.exit_styles.split(",") if s.strip()]

    print(f"Fetching S&P500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Tickers: {len(tickers)}")

    all_trades = []
    total = len(tickers)
    batch_sz = args.batch
    threads = bool(args.threads)

    for bi, batch in enumerate(chunks(tickers, batch_sz), start=1):
        print(f"Batch {bi}: downloading {len(batch)} tickers from {start_dt.date()} to {end_dt.date()} ...")
        try:
            data = yf.download(batch, start=start_dt, end=end_dt, progress=False, group_by="ticker", threads=threads, auto_adjust=False)
        except Exception as e:
            print(f"Batch {bi} download error: {e}")
            continue

        for tkr in batch:
            try:
                sub = slice_from_batch(data, tkr)
                if sub is None or sub.empty or len(sub) < 220:  # need 200d SMA warmup
                    continue
                df = add_indicators(sub.copy()).dropna()
                if df.empty:
                    continue

                # Filter: decent liquidity/price like in your screener (optional tighten here)
                if "Volume" not in df.columns or "Close" not in df.columns:
                    continue
                if df["Volume"].tail(20).mean() < 150_000 or df["Close"].iloc[-1] < 3.0:
                    continue

                # Backtest both screeners
                for scr in ("loose", "strict"):
                    trades = backtest_ticker(df, scr, exit_styles=exit_styles, max_hold=args.max_hold)
                    # attach ticker
                    for tr in trades:
                        tr["Ticker"] = tkr
                    all_trades.extend(trades)

            except Exception as e:
                print(f"  {tkr} error: {e}")
                continue

        # polite throttle
        time.sleep(0.3)

    if not all_trades:
        print("No trades generated. Try widening dates or relaxing filters.")
        sys.exit(0)

    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values(["Screener", "ExitStyle", "EntryDate", "Ticker"], inplace=True)
    trades_df.to_csv("trade_log.csv", index=False)
    print(f"Saved trade_log.csv ({len(trades_df)} rows)")

    summary_df = summarize(trades_df)
    summary_df.to_csv("backtest_summary.csv", index=False)
    print("Saved backtest_summary.csv")
    print("\n=== SUMMARY (top rows) ===")
    print(summary_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
