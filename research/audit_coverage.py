"""Quantify the residual survivorship blind spot and download ETF proxies."""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / 'data'

ETFS = ['SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'SMH', 'IBB', 'ITB', 'KRE', 'XRT', 'OIH', 'JETS', 'RSP']

# Delisted for reasons that destroyed capital vs. reasons that returned capital.
KNOWN_WIPEOUTS = {'LEH', 'FNM', 'FRE', 'ABK', 'JCP', 'SIVB', 'FRC', 'EK', 'ANR',
                  'WIN', 'FTR', 'GGP', 'DNR', 'CHK', 'ENDP', 'MNK', 'BIG', 'AKS',
                  'APOL', 'MWW', 'NOVL', 'JDSU', 'DF', 'RDC', 'ESV', 'QEP', 'WPX'}


def main():
    close = pd.read_parquet(DATA / 'px_close.parquet')
    membership = pd.read_parquet(DATA / 'sp500_membership.parquet')
    tickers = json.loads((DATA / 'universe_tickers.json').read_text())
    current = set(membership.columns[membership.iloc[-1].values])
    removed = [t for t in tickers if t not in current]
    recovered = [t for t in removed if t in close.columns and close[t].dropna().shape[0] > 200]
    blind = sorted(set(removed) - set(recovered))

    print('=' * 78)
    print('SURVIVORSHIP AUDIT')
    print('=' * 78)
    print(f'Tickers ever in the S&P 500 since 2007 : {len(tickers)}')
    print(f'Still members today                    : {len(current)}')
    print(f'Left the index at some point           : {len(removed)}')
    print(f'  ...with usable price history         : {len(recovered)}  '
          f'({100 * len(recovered) / len(removed):.0f}%)')
    print(f'  ...unrecoverable (blind spot)        : {len(blind)}  '
          f'({100 * len(blind) / len(removed):.0f}%)')
    print()
    wipeouts_blind = sorted(KNOWN_WIPEOUTS & set(blind))
    print(f'Known capital-destroying names inside the blind spot ({len(wipeouts_blind)}):')
    print('  ', ', '.join(wipeouts_blind))
    print()
    print('Full blind spot:')
    for i in range(0, len(blind), 14):
        print('  ', ' '.join(f'{t:<7}' for t in blind[i:i + 14]))

    print()
    print('Interpretation: the backtest still cannot see these names, and they are')
    print('skewed toward both bankruptcies (bias UP) and buyouts (bias DOWN).')
    print('Results should be read as an optimistic upper bound, not a forecast.')

    print()
    print('Downloading ETF proxies...')
    df = yf.download(ETFS, start='2006-01-01', end='2026-08-01', interval='1d',
                     progress=False, auto_adjust=True)
    for field in ('Close', 'Open', 'High', 'Low', 'Volume'):
        if field in df:
            df[field].to_parquet(DATA / f'etf_{field.lower()}.parquet')
    ec = df['Close']
    print(f'ETF matrix: {ec.shape[0]} days x {ec.shape[1]}')
    print('First valid date per ETF:')
    print(ec.apply(lambda s: s.first_valid_index()).to_string())


if __name__ == '__main__':
    main()
