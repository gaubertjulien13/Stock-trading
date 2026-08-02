"""Reconstruct point-in-time S&P 500 membership from Wikipedia's change log.

Survivorship bias is the dominant risk for a beaten-down-turnaround strategy: the
companies that actually died get removed from the index, so screening today's members
only ever shows survivors. This walks the historical add/remove table backwards from
the current membership to recover who was in the index on any given date -- including
names that were later deleted.

Output: research/data/sp500_membership.parquet  (date x ticker boolean matrix)
        research/data/universe_tickers.json     (every ticker ever seen)
"""

import json
import re
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / 'data'
OUT.mkdir(exist_ok=True)

WIKI = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
HEADERS = {'User-Agent': 'Mozilla/5.0 (research script)'}

START = pd.Timestamp('2007-01-01')


def _norm(t):
    if not isinstance(t, str):
        return None
    t = t.strip().upper().replace('.', '-')
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    return t or None


def fetch_tables():
    html = requests.get(WIKI, headers=HEADERS, timeout=30).text
    tables = pd.read_html(html)
    current, changes = None, None
    for tb in tables:
        cols = [str(c) for c in tb.columns]
        flat = ' '.join(cols).lower()
        if current is None and 'symbol' in flat and 'gics' in flat:
            current = tb
        if changes is None and 'added' in flat and 'removed' in flat:
            changes = tb
    return current, changes


def build():
    current, changes = fetch_tables()
    if current is None or changes is None:
        raise RuntimeError('Could not locate Wikipedia tables')

    cur_tickers = {_norm(t) for t in current['Symbol']}
    cur_tickers.discard(None)
    print(f'Current S&P 500 members: {len(cur_tickers)}')

    # Flatten the multi-index change table
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

    print(f'Change events on file: {len(ch)}  ({ch["_date"].min().date()} .. {ch["_date"].max().date()})')

    # Walk backwards from today's membership.
    month_ends = pd.date_range(START, pd.Timestamp.today().normalize(), freq='ME')
    members = set(cur_tickers)
    snapshots = {}

    for dt in reversed(month_ends):
        # Undo every change that happened after this snapshot date
        window = ch[ch['_date'] > dt]
        m = set(cur_tickers)
        for _, r in window.iloc[::-1].iterrows():
            if r['_add'] and r['_add'] in m:
                m.discard(r['_add'])      # it was added later -> not a member then
            if r['_rem']:
                m.add(r['_rem'])          # it was removed later -> it was a member then
        snapshots[dt] = m

    all_tickers = sorted(set().union(*snapshots.values()))
    print(f'Distinct tickers ever in universe: {len(all_tickers)}')
    ever_removed = sorted({t for t in all_tickers if t not in cur_tickers})
    print(f'Tickers that left the index (the ones survivorship bias would hide): {len(ever_removed)}')
    print('  sample:', ever_removed[:25])

    mat = pd.DataFrame(
        {t: [t in snapshots[d] for d in month_ends] for t in all_tickers},
        index=month_ends,
    )
    mat.index.name = 'date'
    mat.to_parquet(OUT / 'sp500_membership.parquet')
    (OUT / 'universe_tickers.json').write_text(json.dumps(all_tickers, indent=1))

    print(f'\nMembership matrix: {mat.shape[0]} months x {mat.shape[1]} tickers')
    print('Members per month (sample):')
    print(mat.sum(axis=1).resample('YE').last().to_string())
    return mat


if __name__ == '__main__':
    build()
