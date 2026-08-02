"""Download SEC Form 4 insider transactions and reduce them to open-market buys.

This is the last signal testable without paid data, and the only one in the whole
study that is genuinely point-in-time by construction: a Form 4 has a filing date,
and nobody outside the company could act on it before that date.

Source: SEC "Insider Transactions Data Sets", one ZIP per quarter, 2006Q1 onward.
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

Only transaction code P (open-market purchase) with an acquisition flag is kept.
Everything else -- option exercises, grants, gifts, tax withholding, 10b5-1 sales --
is noise or compensation, not a considered decision to buy at the market price.

Output: research/data/insider_buys.parquet
        one row per (ticker, filing date) with insider count and dollar value
"""

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / 'data'
CACHE = DATA / 'edgar_quarters'
CACHE.mkdir(parents=True, exist_ok=True)

BASE = ('https://www.sec.gov/files/structureddata/data/'
        'insider-transactions-data-sets/{q}_form345.zip')
# SEC requires a descriptive User-Agent with contact details.
HEADERS = {'User-Agent': 'independent-research juliengaubert@example.com'}

QUARTERS = [f'{y}q{q}' for y in range(2006, 2027) for q in (1, 2, 3, 4)]
QUARTERS = QUARTERS[:QUARTERS.index('2026q2')]

WANT = ('SUBMISSION.tsv', 'REPORTINGOWNER.tsv', 'NONDERIV_TRANS.tsv')


def parse_quarter(zf):
    sub = pd.read_csv(zf.open('SUBMISSION.tsv'), sep='\t', dtype=str,
                      usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE',
                               'ISSUERTRADINGSYMBOL'])
    sub = sub[sub['DOCUMENT_TYPE'] == '4']
    sub = sub.dropna(subset=['ISSUERTRADINGSYMBOL'])

    trans = pd.read_csv(zf.open('NONDERIV_TRANS.tsv'), sep='\t', dtype=str,
                        usecols=['ACCESSION_NUMBER', 'TRANS_CODE', 'TRANS_SHARES',
                                 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD'])
    trans = trans[(trans['TRANS_CODE'] == 'P') &
                  (trans['TRANS_ACQUIRED_DISP_CD'] == 'A')]
    if trans.empty:
        return None

    trans['shares'] = pd.to_numeric(trans['TRANS_SHARES'], errors='coerce')
    trans['price'] = pd.to_numeric(trans['TRANS_PRICEPERSHARE'], errors='coerce')
    trans = trans[(trans['shares'] > 0) & (trans['price'] > 0)]
    trans['value'] = trans['shares'] * trans['price']

    own = pd.read_csv(zf.open('REPORTINGOWNER.tsv'), sep='\t', dtype=str,
                      usecols=['ACCESSION_NUMBER', 'RPTOWNERCIK',
                               'RPTOWNER_RELATIONSHIP'])
    rel = own['RPTOWNER_RELATIONSHIP'].fillna('')
    own['is_officer'] = rel.str.contains('Officer', case=False)
    own['is_director'] = rel.str.contains('Director', case=False)
    own = own.groupby('ACCESSION_NUMBER').agg(
        n_owners=('RPTOWNERCIK', 'nunique'),
        officer=('is_officer', 'max'),
        director=('is_director', 'max')).reset_index()

    agg = trans.groupby('ACCESSION_NUMBER').agg(
        shares=('shares', 'sum'), value=('value', 'sum')).reset_index()

    df = sub.merge(agg, on='ACCESSION_NUMBER').merge(own, on='ACCESSION_NUMBER',
                                                     how='left')
    df['filing_date'] = pd.to_datetime(df['FILING_DATE'], format='%d-%b-%Y',
                                       errors='coerce')
    df = df.dropna(subset=['filing_date'])
    df['ticker'] = (df['ISSUERTRADINGSYMBOL'].str.strip().str.upper()
                    .str.replace('.', '-', regex=False))
    df = df[df['ticker'].str.match(r'^[A-Z][A-Z0-9\-]{0,6}$').fillna(False)]

    out = df.groupby(['ticker', 'filing_date']).agg(
        n_filings=('ACCESSION_NUMBER', 'nunique'),
        n_insiders=('n_owners', 'sum'),
        n_officer_filings=('officer', 'sum'),
        n_director_filings=('director', 'sum'),
        shares=('shares', 'sum'),
        value=('value', 'sum')).reset_index()
    return out


def main():
    frames = []
    for q in QUARTERS:
        cached = CACHE / f'{q}.parquet'
        if cached.exists():
            frames.append(pd.read_parquet(cached))
            continue
        url = BASE.format(q=q)
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
        except Exception as e:
            print(f'  {q}: download failed ({e})', flush=True)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                names = set(zf.namelist())
                if not set(WANT) <= names:
                    print(f'  {q}: unexpected archive contents', flush=True)
                    continue
                out = parse_quarter(zf)
        except Exception as e:
            print(f'  {q}: parse failed ({e})', flush=True)
            continue
        if out is None or out.empty:
            print(f'  {q}: no purchases', flush=True)
            continue
        out.to_parquet(cached)
        frames.append(out)
        print(f'  {q}: {len(out):>6,} ticker-days, '
              f'${out["value"].sum() / 1e9:,.1f}B bought', flush=True)
        time.sleep(0.2)

    all_ = pd.concat(frames, ignore_index=True)
    all_ = all_.groupby(['ticker', 'filing_date'], as_index=False).sum()
    all_ = all_.sort_values(['filing_date', 'ticker'])
    all_.to_parquet(DATA / 'insider_buys.parquet')

    print()
    print('=' * 78)
    print('INSIDER OPEN-MARKET PURCHASES')
    print('=' * 78)
    print(f'Rows (ticker x filing date) : {len(all_):,}')
    print(f'Distinct tickers            : {all_["ticker"].nunique():,}')
    print(f'Date range                  : {all_["filing_date"].min().date()} .. '
          f'{all_["filing_date"].max().date()}')
    print(f'Total value                 : ${all_["value"].sum() / 1e9:,.1f}B')
    print()
    print('Purchase filings per year:')
    print(all_.groupby(all_['filing_date'].dt.year)['n_filings'].sum().to_string())


if __name__ == '__main__':
    main()
