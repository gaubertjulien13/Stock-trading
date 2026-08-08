"""Reconstruct the daily market regime for a date range using the live script's
exact classification logic, to tell whether a quiet inbox was the kill switch
or a dead scanner."""
import numpy as np
import pandas as pd
import yfinance as yf

from myutils import get_sp500_tickers

START, END = '2026-07-23', '2026-08-01'

spy = yf.download('SPY', start='2025-06-01', end='2026-08-02', interval='1d', progress=False)
spy_close = spy['Close'].squeeze()

tickers = get_sp500_tickers()
print(f"downloading daily bars for {len(tickers)} tickers (for breadth)...")
closes = {}
for i in range(0, len(tickers), 60):
    batch = tickers[i:i + 60]
    d = yf.download(batch, start='2025-06-01', end='2026-08-02', interval='1d',
                    progress=False, group_by='ticker', threads=True)
    for t in batch:
        try:
            sub = d[t]['Close'].dropna() if isinstance(d.columns, pd.MultiIndex) else d['Close'].dropna()
            if len(sub) > 200:
                closes[t] = sub
        except Exception:
            pass
print(f"got {len(closes)} tickers with enough history\n")

panel = pd.DataFrame(closes)
sma200_panel = panel.rolling(200).mean()
above200 = panel > sma200_panel

sma50 = spy_close.rolling(50).mean()
sma200 = spy_close.rolling(200).mean()

days = spy_close.loc[START:END].index
print(f"{'date':12s} {'SPY':>8} {'5d%':>7} {'SMA50':>7} {'SMA200':>7} {'breadth':>8}  REGIME")
print("-" * 68)
for d in days:
    px = float(spy_close.loc[d])
    pos = spy_close.index.get_loc(d)
    ret5 = float((spy_close.iloc[pos] / spy_close.iloc[pos - 5] - 1) * 100) if pos >= 5 else 0.0
    a50 = px > float(sma50.loc[d])
    a200 = px > float(sma200.loc[d])
    row = above200.loc[d].dropna()
    breadth = row.mean() * 100 if len(row) else np.nan

    if (not a200) or ret5 < -5.0:
        regime = 'RISK_OFF'
    elif (not a50) or breadth < 40.0 or ret5 < -2.0:
        regime = 'CAUTION'
    else:
        regime = 'RISK_ON'
    trigger = ""
    if regime == 'CAUTION':
        why = []
        if not a50:
            why.append("SPY<SMA50")
        if breadth < 40:
            why.append(f"breadth {breadth:.0f}%")
        if ret5 < -2:
            why.append(f"5d {ret5:.1f}%")
        trigger = "  <- " + ", ".join(why)
    print(f"{d.date()!s:12s} {px:8.2f} {ret5:+7.2f} {'above' if a50 else 'BELOW':>7} "
          f"{'above' if a200 else 'BELOW':>7} {breadth:7.1f}%  {regime}{trigger}")
