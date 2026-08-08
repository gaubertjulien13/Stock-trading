"""Point-in-time fundamentals from SEC XBRL company facts.

PLAYBOOK.md flags this as the main missing input: yfinance returns *today's*
balance sheet, so anything built on it leaks the future and cannot be validated.
The SEC's XBRL API carries a `filed` date on every fact, so a series can be
reconstructed as it was knowable on any past date - which makes the pillar-3
("damage is temporary") and pillar-6 ("survivability") questions answerable with
evidence instead of a price proxy.

Free, no key, no paid feed. SEC asks for a descriptive User-Agent with a contact
address and rate-limits to 10 requests/second; both are honored below.

    venv/bin/python3 research/fetch_fundamentals.py --tickers BSX,INTU,INTC
    venv/bin/python3 research/fetch_fundamentals.py --from-watchlist --top 15

Set a contact address once (SEC blocks generic agents):
    export SEC_USER_AGENT="Your Name your@email.com"
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from engine import DATA

FUND_DIR = DATA / 'fundamentals'
TICKER_MAP = DATA / 'sec_ticker_cik.json'
SEC_RATE_SLEEP = 0.12          # ~8 req/s, under the published 10/s ceiling
QUARTER_DAYS = (75, 115)
ANNUAL_DAYS = (330, 400)

# Companies tag the same economic concept differently; first match wins.
CONCEPTS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet'],
    'gross_profit': ['GrossProfit'],
    'cost_of_revenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfSales'],
    'operating_income': ['OperatingIncomeLoss'],
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
    'rnd': ['ResearchAndDevelopmentExpense'],
    'sga': ['SellingGeneralAndAdministrativeExpense',
            'GeneralAndAdministrativeExpense'],
    'cfo': ['NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'capex': ['PaymentsToAcquirePropertyPlantAndEquipment',
              'PaymentsToAcquireProductiveAssets'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue',
             'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'assets': ['Assets'],
    'liabilities': ['Liabilities'],
    'equity': ['StockholdersEquity',
               'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'debt_lt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
    'debt_cur': ['LongTermDebtCurrent', 'DebtCurrent'],
    'shares': ['CommonStockSharesOutstanding', 'CommonStockSharesIssued'],
}
INSTANT = {'cash', 'assets', 'liabilities', 'equity', 'debt_lt', 'debt_cur', 'shares'}


def _user_agent():
    ua = os.environ.get('SEC_USER_AGENT')
    if ua:
        return ua
    # Reuse the address already configured for alert email, if present.
    env = Path(__file__).resolve().parent.parent / '.stock_screener.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('ALERT_FROM_EMAIL'):
                addr = line.split('=', 1)[1].strip().strip('"\'')
                if addr:
                    return f'finance-research {addr}'
    return 'finance-research contact@example.com'


def _session():
    s = requests.Session()
    s.headers.update({'User-Agent': _user_agent(),
                      'Accept-Encoding': 'gzip, deflate'})
    return s


def load_ticker_cik(sess, refresh=False):
    """SEC's ticker -> CIK directory (needed to address the facts API)."""
    if TICKER_MAP.exists() and not refresh:
        return json.loads(TICKER_MAP.read_text())
    r = sess.get('https://www.sec.gov/files/company_tickers.json', timeout=30)
    r.raise_for_status()
    raw = r.json()
    out = {v['ticker'].upper(): f"{int(v['cik_str']):010d}" for v in raw.values()}
    TICKER_MAP.parent.mkdir(parents=True, exist_ok=True)
    TICKER_MAP.write_text(json.dumps(out))
    return out


def fetch_facts(ticker, cik, sess, refresh=False):
    path = FUND_DIR / f'{ticker}.json'
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    r = sess.get(url, timeout=45)
    time.sleep(SEC_RATE_SLEEP)
    if r.status_code != 200:
        return None
    data = r.json()
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _concept_series(facts, names, instant):
    """Tidy one concept into a frame, keeping `filed` so it stays point-in-time."""
    gaap = facts.get('facts', {}).get('us-gaap', {})
    for name in names:
        node = gaap.get(name)
        if not node:
            continue
        for unit, entries in node.get('units', {}).items():
            if unit not in ('USD', 'shares'):
                continue
            rows = []
            for e in entries:
                if 'end' not in e or 'val' not in e:
                    continue
                end = pd.Timestamp(e['end'])
                filed = pd.Timestamp(e.get('filed', e['end']))
                if instant:
                    rows.append({'end': end, 'start': pd.NaT, 'val': float(e['val']),
                                 'filed': filed, 'form': e.get('form', ''), 'days': 0})
                else:
                    if 'start' not in e:
                        continue
                    start = pd.Timestamp(e['start'])
                    rows.append({'end': end, 'start': start, 'val': float(e['val']),
                                 'filed': filed, 'form': e.get('form', ''),
                                 'days': (end - start).days})
            if rows:
                df = pd.DataFrame(rows).sort_values(['end', 'filed'])
                # Restatements: keep the first filing of each period (what was known then)
                return df.drop_duplicates(subset=['end', 'days'],
                                          keep='first').reset_index(drop=True), name
    return None, None


def _discrete_quarters(df):
    """De-cumulate year-to-date filings into discrete quarters.

    Many filers report cumulative YTD figures (Q1=89d, H1=180d, 9M=272d, FY=364d)
    instead of discrete quarters. Entries sharing a fiscal-year start form a
    cumulative ladder, so differencing successive rungs recovers the real quarter.
    Filers who already report discrete quarters pass through untouched, since each
    of their entries sits alone in its start-group.
    """
    empty = pd.DataFrame(columns=['end', 'val', 'filed', 'days'])
    if df is None or df.empty:
        return empty
    d = df.dropna(subset=['start'])
    if d.empty:
        return empty
    rows = []
    for _, g in d.groupby('start'):
        g = g.sort_values('days')
        prev_val, prev_days = 0.0, 0
        for _, r in g.iterrows():
            val, days = r['val'] - prev_val, r['days'] - prev_days
            if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
                rows.append({'end': r['end'], 'val': val, 'filed': r['filed'], 'days': days})
            prev_val, prev_days = r['val'], r['days']
    if not rows:
        return empty
    out = pd.DataFrame(rows).sort_values('end')
    return out.drop_duplicates(subset='end', keep='first').reset_index(drop=True)


def _ttm_series(df, asof=None):
    """Trailing-twelve-month value at each quarter end.

    Requires four contiguous quarters covering ~a year, so a gap in the filing
    history produces no TTM rather than a silently wrong one.
    """
    q = _discrete_quarters(df)
    if asof is not None and not q.empty:
        q = q[q['filed'] <= pd.Timestamp(asof)]
    if len(q) < 4:
        return pd.DataFrame(columns=['end', 'val'])
    rows = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3:i + 1]
        if 330 <= w['days'].sum() <= 400 and 240 <= (w['end'].iloc[-1] - w['end'].iloc[0]).days <= 300:
            rows.append({'end': w['end'].iloc[-1], 'val': float(w['val'].sum())})
    return pd.DataFrame(rows)


def _ttm_at(ts, quarters_back=0):
    """TTM value `quarters_back` quarters before the latest reading."""
    if ts is None or ts.empty:
        return np.nan
    i = len(ts) - 1 - quarters_back
    return float(ts['val'].iloc[i]) if i >= 0 else np.nan


def _latest_instant(df, asof=None):
    if df is None or df.empty:
        return np.nan, None
    if asof is not None:
        df = df[df['filed'] <= pd.Timestamp(asof)]
        if df.empty:
            return np.nan, None
    df = df.sort_values('end')
    return float(df['val'].iloc[-1]), df['end'].iloc[-1]


def _value_near(df, target, tol_days=120):
    """Instant value closest to a target date - used for like-for-like YoY."""
    if df is None or df.empty:
        return np.nan
    d = df.dropna(subset=['end'])
    if d.empty:
        return np.nan
    gap = (d['end'] - target).abs()
    if gap.min() > pd.Timedelta(days=tol_days):
        return np.nan
    return float(d.loc[gap.idxmin(), 'val'])


def _pct(new, old):
    if not (np.isfinite(new) and np.isfinite(old)) or old == 0:
        return np.nan
    return (new / abs(old) - 1.0) * 100.0


def summarize(ticker, facts, asof=None):
    """Turn raw XBRL into the handful of numbers a turnaround thesis rests on."""
    if not facts:
        return None
    series = {}
    for key, names in CONCEPTS.items():
        df, tag = _concept_series(facts, names, key in INSTANT)
        series[key] = df

    ttm = {k: _ttm_series(series[k], asof) for k in
           ('revenue', 'gross_profit', 'cost_of_revenue', 'operating_income',
            'net_income', 'rnd', 'sga', 'cfo', 'capex')}

    rev_ttm = _ttm_at(ttm['revenue'])
    rev_prev = _ttm_at(ttm['revenue'], 4)
    rev_3y = _ttm_at(ttm['revenue'], 12)
    rev_end = ttm['revenue']['end'].iloc[-1] if not ttm['revenue'].empty else None

    gp_ttm, gp_prev = _ttm_at(ttm['gross_profit']), _ttm_at(ttm['gross_profit'], 4)
    if not np.isfinite(gp_ttm):  # not all filers tag GrossProfit; derive it
        cor, cor_prev = _ttm_at(ttm['cost_of_revenue']), _ttm_at(ttm['cost_of_revenue'], 4)
        if np.isfinite(cor) and np.isfinite(rev_ttm):
            gp_ttm = rev_ttm - cor
        if np.isfinite(cor_prev) and np.isfinite(rev_prev):
            gp_prev = rev_prev - cor_prev

    op_ttm, op_prev = _ttm_at(ttm['operating_income']), _ttm_at(ttm['operating_income'], 4)
    if not np.isfinite(op_ttm):  # e.g. NKE reports no OperatingIncomeLoss tag
        def _derive_op(gp, back):
            sga, rd = _ttm_at(ttm['sga'], back), _ttm_at(ttm['rnd'], back)
            if not (np.isfinite(gp) and np.isfinite(sga)):
                return np.nan
            return gp - sga - (rd if np.isfinite(rd) else 0.0)
        op_ttm, op_prev = _derive_op(gp_ttm, 0), _derive_op(gp_prev, 4)
    ni_ttm = _ttm_at(ttm['net_income'])
    rnd_ttm = _ttm_at(ttm['rnd'])
    cfo_ttm = _ttm_at(ttm['cfo'])
    capex_ttm = _ttm_at(ttm['capex'])

    cash, cash_end = _latest_instant(series['cash'], asof)
    assets, _ = _latest_instant(series['assets'], asof)
    liab, _ = _latest_instant(series['liabilities'], asof)
    equity, _ = _latest_instant(series['equity'], asof)
    dlt, _ = _latest_instant(series['debt_lt'], asof)
    dcur, _ = _latest_instant(series['debt_cur'], asof)
    sh, sh_end = _latest_instant(series['shares'], asof)
    sh_prev = (_value_near(series['shares'], sh_end - pd.Timedelta(days=365))
               if sh_end is not None else np.nan)

    fcf = (cfo_ttm - capex_ttm) if (np.isfinite(cfo_ttm) and np.isfinite(capex_ttm)) else np.nan
    debt = np.nansum([d for d in (dlt, dcur) if np.isfinite(d)]) or np.nan
    net_debt = (debt - cash) if (np.isfinite(debt) and np.isfinite(cash)) else np.nan

    def margin(num, den):
        return (num / den * 100.0) if (np.isfinite(num) and np.isfinite(den) and den) else np.nan

    out = {
        'ticker': ticker,
        'asof': str(asof.date()) if isinstance(asof, pd.Timestamp) else (asof or 'latest'),
        'period_end': str(rev_end.date()) if rev_end is not None else None,
        'revenue_ttm': rev_ttm,
        'revenue_yoy_pct': _pct(rev_ttm, rev_prev),
        'revenue_3y_pct': _pct(rev_ttm, rev_3y),
        'gross_margin_pct': margin(gp_ttm, rev_ttm),
        'gross_margin_delta_pp': (margin(gp_ttm, rev_ttm) - margin(gp_prev, rev_prev)
                                  if np.isfinite(gp_prev) and np.isfinite(rev_prev) else np.nan),
        'operating_margin_pct': margin(op_ttm, rev_ttm),
        'operating_margin_delta_pp': (margin(op_ttm, rev_ttm) - margin(op_prev, rev_prev)
                                      if np.isfinite(op_prev) and np.isfinite(rev_prev) else np.nan),
        'net_income_ttm': ni_ttm,
        'rnd_pct_revenue': margin(rnd_ttm, rev_ttm),
        'fcf_ttm': fcf,
        'fcf_margin_pct': margin(fcf, rev_ttm),
        'cash': cash,
        'total_debt': debt,
        'net_debt': net_debt,
        'net_debt_to_fcf': (net_debt / fcf) if (np.isfinite(net_debt) and np.isfinite(fcf)
                                                and fcf > 0) else np.nan,
        'equity': equity,
        'assets': assets,
        'liabilities': liab,
        'shares_out': sh,
        'dilution_yoy_pct': _pct(sh, sh_prev),
        'balance_date': str(cash_end.date()) if cash_end is not None else None,
    }
    out['flags'] = _flags(out)
    return out


def _flags(s):
    """Plain-language read on the numbers, so the brief does not just restate them.

    These are descriptive, not predictive - nothing here has been validated as a
    return signal, and it should not be treated as one.
    """
    f = []
    rev, om, fcf = s['revenue_yoy_pct'], s['operating_margin_delta_pp'], s['fcf_ttm']
    if np.isfinite(rev):
        f.append('revenue growing' if rev > 3 else
                 'revenue declining' if rev < -3 else 'revenue flat')
    if np.isfinite(om):
        f.append('margins expanding' if om > 1 else
                 'margins compressing' if om < -1 else 'margins stable')
    if np.isfinite(fcf):
        f.append('FCF positive' if fcf > 0 else 'FCF NEGATIVE')
    nd, netdebt = s['net_debt_to_fcf'], s['net_debt']
    if np.isfinite(netdebt) and netdebt < 0:
        f.append(f'net cash {abs(netdebt) / 1e9:,.1f}B')
    elif np.isfinite(nd):
        f.append(f'net debt {nd:.1f}x FCF' + (' (heavy)' if nd > 5 else ''))
    d = s['dilution_yoy_pct']
    if np.isfinite(d):
        if d > 3:
            f.append(f'diluting {d:.1f}%/yr')
        elif d < -1:
            f.append(f'buying back {abs(d):.1f}%/yr')
    if np.isfinite(s['rnd_pct_revenue']) and s['rnd_pct_revenue'] > 10:
        f.append(f"R&D {s['rnd_pct_revenue']:.0f}% of revenue")
    return f


def get(tickers, refresh=False, asof=None, quiet=False):
    sess = _session()
    cikmap = load_ticker_cik(sess, refresh=refresh)
    out = {}
    for i, t in enumerate(tickers, 1):
        cik = cikmap.get(t.upper())
        if not cik:
            if not quiet:
                print(f"  {t}: no CIK in SEC directory")
            continue
        try:
            facts = fetch_facts(t.upper(), cik, sess, refresh=refresh)
            s = summarize(t.upper(), facts, asof=asof)
        except Exception as exc:
            if not quiet:
                print(f"  {t}: {type(exc).__name__} {exc}")
            continue
        if s:
            out[t.upper()] = s
        if not quiet:
            print(f"\r  fundamentals {i}/{len(tickers)}  {t:6s}", end='', flush=True)
    if not quiet:
        print("\r" + " " * 40 + "\r", end='')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tickers', help='Comma-separated tickers')
    ap.add_argument('--refresh', action='store_true', help='Ignore cache')
    args = ap.parse_args()

    if not args.tickers:
        ap.error('--tickers is required')
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]

    print(f"SEC User-Agent: {_user_agent()}")
    data = get(tickers, refresh=args.refresh)
    for t, s in data.items():
        print(f"\n=== {t} ===  period end {s['period_end']}  balance {s['balance_date']}")
        print(f"  revenue TTM   ${s['revenue_ttm'] / 1e9:,.2f}B   "
              f"YoY {s['revenue_yoy_pct']:+.1f}%   3y {s['revenue_3y_pct']:+.1f}%")
        print(f"  gross margin  {s['gross_margin_pct']:.1f}%  ({s['gross_margin_delta_pp']:+.1f}pp YoY)")
        print(f"  op margin     {s['operating_margin_pct']:.1f}%  ({s['operating_margin_delta_pp']:+.1f}pp YoY)")
        print(f"  FCF TTM       ${s['fcf_ttm'] / 1e9:,.2f}B   margin {s['fcf_margin_pct']:.1f}%")
        print(f"  cash ${s['cash'] / 1e9:,.2f}B   debt ${s['total_debt'] / 1e9:,.2f}B   "
              f"net debt ${s['net_debt'] / 1e9:,.2f}B")
        print(f"  flags: {', '.join(s['flags'])}")


if __name__ == '__main__':
    main()
