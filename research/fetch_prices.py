"""Download and cache daily OHLCV for the full point-in-time universe.

Includes delisted tickers. How many of those actually return data determines how
much survivorship bias remains in the backtest, so the coverage report matters as
much as the data.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / 'data'
DATA.mkdir(exist_ok=True)

START = '2006-01-01'
# yfinance treats `end` as exclusive, so reach past today to include the latest close.
END = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
BATCH = 40

# Sector / industry proxies, plus the market benchmark.
ETFS = ['SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'SMH', 'IBB', 'ITB', 'KRE', 'XRT', 'OIH', 'JETS', 'OEF']


def download(tickers, label):
    out = {}
    failed = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        try:
            df = yf.download(chunk, start=START, end=END, interval='1d',
                             progress=False, auto_adjust=True, threads=True,
                             group_by='column')
        except Exception as e:
            print(f'  batch {i} error: {e}')
            failed.extend(chunk)
            continue

        if df is None or df.empty:
            failed.extend(chunk)
            continue

        for field in ('Close', 'Open', 'High', 'Low', 'Volume'):
            if field not in df:
                continue
            sub = df[field]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(chunk[0])
            out.setdefault(field, []).append(sub)

        got = set(df['Close'].columns) if 'Close' in df else set()
        missing = [t for t in chunk if t not in got or df['Close'][t].dropna().empty]
        failed.extend(missing)
        print(f'  {label} {i + len(chunk)}/{len(tickers)}  (cumulative missing: {len(failed)})',
              flush=True)
        time.sleep(0.4)

    merged = {}
    for field, frames in out.items():
        m = pd.concat(frames, axis=1)
        m = m.loc[:, ~m.columns.duplicated()]
        merged[field] = m.sort_index()
    return merged, failed


def main():
    tickers = json.loads((DATA / 'universe_tickers.json').read_text())
    print(f'Universe: {len(tickers)} tickers (incl. delisted)')

    merged, failed = download(tickers, 'stocks')

    for field, m in merged.items():
        m.to_parquet(DATA / f'px_{field.lower()}.parquet')

    close = merged['Close']
    print(f'\nPrice matrix: {close.shape[0]} days x {close.shape[1]} tickers')
    print(f'No data returned for {len(failed)} tickers')

    membership = pd.read_parquet(DATA / 'sp500_membership.parquet')
    current = set(membership.columns[membership.iloc[-1].values])
    removed = [t for t in tickers if t not in current]
    removed_with_data = [t for t in removed if t in close.columns
                         and close[t].dropna().shape[0] > 200]
    print(f'\nSURVIVORSHIP CHECK')
    print(f'  Tickers that left the index: {len(removed)}')
    print(f'  ...of which we recovered usable history: {len(removed_with_data)} '
          f'({100 * len(removed_with_data) / max(len(removed), 1):.0f}%)')
    print(f'  Unrecoverable (true blind spot): {len(removed) - len(removed_with_data)}')

    (DATA / 'missing_tickers.json').write_text(json.dumps(sorted(set(failed)), indent=1))

    etf, etf_failed = download(ETFS, 'etfs')
    for field, m in etf.items():
        m.to_parquet(DATA / f'etf_{field.lower()}.parquet')
    print(f'\nETFs downloaded: {etf["Close"].shape[1]}, failed: {etf_failed}')


if __name__ == '__main__':
    main()
