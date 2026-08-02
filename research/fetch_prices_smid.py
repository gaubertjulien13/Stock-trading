"""Download daily OHLCV for the mid/small-cap universe, including delisted names.

Only the tickers not already present in the S&P 500 pull are fetched. As with the
large-cap run, the coverage report matters as much as the data: whatever fails to
download is a name the backtest cannot see, and delisted names fail most often.

Output: research/data/px_smid_{close,open,high,low,volume}.parquet
        research/data/etf_smid_close.parquet   (IJH / IJR size benchmarks)
"""

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / 'data'

START = '2010-01-01'
END = '2026-08-01'
BATCH = 50

BENCH = ['IJH', 'IJR', 'MDY', 'SPY', 'RSP', 'IWM']


def download(tickers, label):
    out, failed = {}, []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        try:
            df = yf.download(chunk, start=START, end=END, interval='1d',
                             progress=False, auto_adjust=True, threads=True,
                             group_by='column')
        except Exception as e:
            print(f'  batch {i} error: {e}', flush=True)
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
        print(f'  {label} {min(i + BATCH, len(tickers))}/{len(tickers)}  '
              f'(cumulative missing: {len(failed)})', flush=True)
        time.sleep(0.3)

    merged = {}
    for field, frames in out.items():
        m = pd.concat(frames, axis=1)
        m = m.loc[:, ~m.columns.duplicated()]
        merged[field] = m.sort_index()
    return merged, failed


def main():
    universe = set(json.loads((DATA / 'universe_smid.json').read_text()))
    have = set(pd.read_parquet(DATA / 'px_close.parquet').columns)
    todo = sorted(universe - have)
    print(f'Mid/small universe: {len(universe)}   already downloaded: {len(universe & have)}   '
          f'to fetch: {len(todo)}')

    merged, failed = download(todo, 'smid')
    for field, m in merged.items():
        m.to_parquet(DATA / f'px_smid_{field.lower()}.parquet')

    close = merged['Close']
    print(f'\nPrice matrix: {close.shape[0]} days x {close.shape[1]} tickers')
    print(f'No data returned for {len(failed)} tickers')

    sp400 = pd.read_parquet(DATA / 'sp400_membership.parquet')
    sp600 = pd.read_parquet(DATA / 'sp600_membership.parquet')
    print()
    print('=' * 78)
    print('SURVIVORSHIP AUDIT  --  mid/small caps')
    print('=' * 78)
    for name, mat in (('S&P 400', sp400), ('S&P 600', sp600)):
        cur = set(mat.columns[mat.iloc[-1].values])
        ever = set(mat.columns)
        removed = ever - cur
        rec = {t for t in removed
               if (t in close.columns and close[t].dropna().shape[0] > 200)
               or t in have}
        blind = removed - rec
        print(f'{name}: ever {len(ever)}, today {len(cur)}, left {len(removed)}, '
              f'recovered {len(rec)} ({100 * len(rec) / max(len(removed), 1):.0f}%), '
              f'blind {len(blind)}')

    (DATA / 'missing_smid.json').write_text(json.dumps(sorted(set(failed)), indent=1))

    etf, ef = download(BENCH, 'bench')
    for field, m in etf.items():
        m.to_parquet(DATA / f'etf_smid_{field.lower()}.parquet')
    print(f'\nBenchmarks: {list(etf["Close"].columns)}  failed: {ef}')
    print('First valid date per benchmark:')
    print(etf['Close'].apply(lambda s: s.first_valid_index()).to_string())


if __name__ == '__main__':
    main()
