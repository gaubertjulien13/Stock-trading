"""Point-in-time membership for the S&P MidCap 400 and SmallCap 600.

Same reconstruction as build_universe.py, but run for the two smaller indices,
because the large-cap null result says nothing about where these effects are
supposed to live. The change log is walked backwards from today's membership.

The critical output is not the matrix but the honesty check: a change log that
starts in 2015 cannot reconstruct 2008, and pretending otherwise silently
reintroduces exactly the survivorship bias the whole exercise is trying to avoid.

Output: research/data/{sp400,sp600}_membership.parquet
        research/data/universe_smid.json
"""

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / 'data'
OUT.mkdir(exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (research script)'}
START = pd.Timestamp('2007-01-01')

INDICES = {
    'sp400': 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
    'sp600': 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies',
}


def _norm(t):
    if not isinstance(t, str):
        return None
    t = t.strip().upper().replace('.', '-')
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    return t or None


def fetch_tables(url):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    tables = pd.read_html(StringIO(html))
    current, changes = None, None
    for tb in tables:
        flat = ' '.join(str(c) for c in tb.columns).lower()
        if current is None and 'symbol' in flat and 'gics' in flat:
            current = tb
        if changes is None and 'added' in flat and 'removed' in flat:
            changes = tb
    return current, changes


def build(name, url):
    current, changes = fetch_tables(url)
    if current is None or changes is None:
        raise RuntimeError(f'{name}: could not locate Wikipedia tables')

    cur = {_norm(t) for t in current['Symbol']}
    cur.discard(None)

    ch = changes.copy()
    if isinstance(ch.columns, pd.MultiIndex):
        ch.columns = [' '.join(str(x) for x in c).strip() for c in ch.columns]
    date_col = next(c for c in ch.columns if 'date' in c.lower())
    add_col = next(c for c in ch.columns if 'added' in c.lower() and 'ticker' in c.lower())
    rem_col = next(c for c in ch.columns if 'removed' in c.lower() and 'ticker' in c.lower())

    ch['_date'] = pd.to_datetime(ch[date_col], errors='coerce')
    ch = ch.dropna(subset=['_date']).sort_values('_date')
    ch['_add'] = ch[add_col].map(_norm)
    ch['_rem'] = ch[rem_col].map(_norm)

    print(f'\n=== {name.upper()}')
    print(f'  members today          : {len(cur)}')
    print(f'  change events on file  : {len(ch)}')
    print(f'  log covers             : {ch["_date"].min().date()} .. {ch["_date"].max().date()}')

    per_year = ch.groupby(ch['_date'].dt.year).size()
    print('  events per year:')
    print('   ', per_year.to_dict())

    month_ends = pd.date_range(START, pd.Timestamp.today().normalize(), freq='ME')
    snapshots = {}
    for dt in reversed(month_ends):
        window = ch[ch['_date'] > dt]
        m = set(cur)
        for _, r in window.iloc[::-1].iterrows():
            if r['_add'] and r['_add'] in m:
                m.discard(r['_add'])
            if r['_rem']:
                m.add(r['_rem'])
        snapshots[dt] = m

    all_t = sorted(set().union(*snapshots.values()))
    mat = pd.DataFrame({t: [t in snapshots[d] for d in month_ends] for t in all_t},
                       index=month_ends)
    mat.index.name = 'date'
    mat.to_parquet(OUT / f'{name}_membership.parquet')

    counts = mat.sum(axis=1)
    print(f'  distinct tickers ever  : {len(all_t)}')
    print(f'  left the index         : {len([t for t in all_t if t not in cur])}')
    print('  reconstructed members per year (should sit near the index size):')
    print('   ', counts.resample('YE').last().astype(int).to_dict())
    return mat, all_t


def main():
    every = set()
    mats = {}
    for name, url in INDICES.items():
        mat, tickers = build(name, url)
        mats[name] = mat
        every |= set(tickers)

    (OUT / 'universe_smid.json').write_text(json.dumps(sorted(every), indent=1))
    print(f'\nCombined mid+small universe: {len(every)} tickers')

    sp500 = pd.read_parquet(OUT / 'sp500_membership.parquet')
    overlap = every & set(sp500.columns)
    print(f'Overlap with the S&P 500 universe already downloaded: {len(overlap)}')
    print(f'New tickers to fetch: {len(every - set(sp500.columns))}')

    print()
    print('=' * 78)
    print('HONESTY CHECK')
    print('=' * 78)
    print('Where the reconstructed member count drifts far below the true index size,')
    print('the change log is incomplete and those years cannot be used. Read the per-')
    print('year counts above before trusting any backtest window.')


if __name__ == '__main__':
    main()
