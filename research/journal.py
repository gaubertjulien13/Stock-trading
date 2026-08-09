"""Decision journal - the only way this approach can ever be validated.

Pillars 4 and 5 cannot be backtested: there is no historical archive of headlines
or of what a reasonable person would have concluded from them. So the qualitative
layer can only be validated forward, by recording decisions before the outcome is
known and scoring them later against an honest benchmark.

Two benchmarks are used, because beating neither means the work is not paying:
  - SPY over the same holding period (what the money would have done otherwise)
  - the rest of that day's Stage-1 candidate pool (what picking blind would have
    given, which isolates judgment from the screen)

Recording the thesis matters as much as the decision. Without it, hindsight will
rewrite what you believed, and the review becomes theatre.

    venv/bin/python3 research/journal.py add --ticker BSX --decision buy \\
        --thesis "Restructuring plus Q2 beat; recall is contained and one-off" \\
        --mechanism "New management cost program" --conviction 4
    venv/bin/python3 research/journal.py list
    venv/bin/python3 research/journal.py review
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DATA

JOURNAL = DATA / 'decision_journal.csv'
COLUMNS = ['date', 'ticker', 'decision', 'price', 'conviction', 'mechanism',
           'thesis', 'pillars_met', 'size_pct', 'horizon_days', 'notes']
DECISIONS = ('buy', 'watch', 'pass')


def load():
    if not JOURNAL.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(JOURNAL)
    # 'mixed' tolerates files written before dates were normalised on save.
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df


def save(df):
    """Dates are written date-only so the file never gains mixed formats.

    Round-tripping a parsed Timestamp writes '2026-08-09 00:00:00' while a fresh
    entry writes '2026-08-09'. Once both are present, reading the file back
    raises, which surfaces as the journal breaking on the third entry.
    """
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'], format='mixed').dt.strftime('%Y-%m-%d')
    out.to_csv(JOURNAL, index=False)


def _price_now(ticker):
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period='5d', auto_adjust=True)
        return float(h['Close'].iloc[-1]) if len(h) else np.nan
    except Exception:
        return np.nan


def cmd_add(args):
    df = load()
    price = args.price if args.price else _price_now(args.ticker.upper())
    row = {
        'date': args.date or datetime.now().strftime('%Y-%m-%d'),
        'ticker': args.ticker.upper(),
        'decision': args.decision,
        'price': price,
        'conviction': args.conviction,
        'mechanism': args.mechanism or '',
        'thesis': args.thesis,
        'pillars_met': args.pillars or '',
        'size_pct': args.size if args.size is not None else '',
        'horizon_days': args.horizon,
        'notes': args.notes or '',
    }
    if args.decision == 'buy' and not args.mechanism:
        print("  warning: a buy without a named recovery mechanism is the pattern "
              "the analog study says fails. Recording anyway.")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    print(f"  logged {row['decision'].upper()} {row['ticker']} @ "
          f"{'n/a' if not np.isfinite(price) else f'${price:.2f}'} on {row['date']}")
    print(f"  thesis: {row['thesis']}")
    print(f"  {len(df)} decisions on file "
          f"({(df['decision'] == 'buy').sum()} buys). "
          f"{max(0, 20 - len(df))} more before a review is meaningful.")


def cmd_list(args):
    df = load()
    if df.empty:
        print("  no decisions logged yet")
        return
    if args.decision:
        df = df[df['decision'] == args.decision]
    for _, r in df.sort_values('date').iterrows():
        px = f"${r['price']:.2f}" if pd.notna(r['price']) else 'n/a'
        print(f"  {r['date'].date()}  {r['decision']:5s}  {r['ticker']:6s} {px:>9s}  "
              f"conv {r['conviction']}  {str(r['thesis'])[:70]}")
    print(f"\n  {len(df)} decisions")


def _forward(ticker, start, horizon_days):
    """Return (stock_pct, spy_pct) from `start` over the horizon, or to today."""
    try:
        import yfinance as yf
        end = min(pd.Timestamp.now(), pd.Timestamp(start) + pd.Timedelta(days=horizon_days + 5))
        data = yf.download([ticker, 'SPY'], start=pd.Timestamp(start) - pd.Timedelta(days=5),
                           end=end + pd.Timedelta(days=1), progress=False, auto_adjust=True)
        px = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        px = px.dropna()
        if len(px) < 2:
            return np.nan, np.nan
        s = px[px.index >= pd.Timestamp(start)]
        if s.empty or len(s) < 2:
            return np.nan, np.nan
        r = (s.iloc[-1] / s.iloc[0] - 1.0) * 100
        return float(r.get(ticker, np.nan)), float(r.get('SPY', np.nan))
    except Exception:
        return np.nan, np.nan


def cmd_review(args):
    df = load()
    if df.empty:
        print("  no decisions logged yet")
        return
    print(f"  scoring {len(df)} decisions against SPY over the same windows...\n")
    rows = []
    for _, r in df.iterrows():
        stock, spy = _forward(r['ticker'], r['date'], int(r['horizon_days'] or 180))
        rows.append({**r.to_dict(), 'ret_pct': stock, 'spy_pct': spy,
                     'excess_pp': stock - spy if np.isfinite(stock) and np.isfinite(spy) else np.nan})
    res = pd.DataFrame(rows).dropna(subset=['ret_pct'])
    if res.empty:
        print("  no resolvable outcomes yet")
        return

    print(f"  {'date':11s} {'dec':6s} {'ticker':7s} {'return':>8s} {'SPY':>8s} {'excess':>9s}  thesis")
    for _, r in res.sort_values('date').iterrows():
        print(f"  {str(r['date'])[:10]:11s} {r['decision']:6s} {r['ticker']:7s} "
              f"{r['ret_pct']:+7.1f}% {r['spy_pct']:+7.1f}% {r['excess_pp']:+8.1f}pp  "
              f"{str(r['thesis'])[:44]}")

    print()
    for dec in DECISIONS:
        g = res[res['decision'] == dec]
        if g.empty:
            continue
        print(f"  {dec.upper():6s} n={len(g):3d}  mean {g['ret_pct'].mean():+6.2f}%  "
              f"vs SPY {g['excess_pp'].mean():+6.2f}pp  beat SPY {(g['excess_pp'] > 0).mean() * 100:3.0f}%  "
              f"median excess {g['excess_pp'].median():+6.2f}pp")

    buys = res[res['decision'] == 'buy']
    passes = res[res['decision'] == 'pass']
    if not buys.empty and not passes.empty:
        gap = buys['excess_pp'].mean() - passes['excess_pp'].mean()
        print(f"\n  Selection value: buys beat passes by {gap:+.2f}pp on average. "
              f"{'Judgment is separating them.' if gap > 0 else 'No separation yet.'}")
    if len(res) < 20:
        print(f"\n  Only {len(res)} resolved decisions. With a payoff this skewed, "
              f"nothing here is\n  meaningful until roughly 20-30 - treat it as a habit, "
              f"not a verdict.")
    if not buys.empty and buys['conviction'].notna().any():
        c = buys.dropna(subset=['conviction'])
        if len(c) >= 5 and c['conviction'].nunique() > 1:
            corr = c['conviction'].corr(c['excess_pp'], method='spearman')
            print(f"  Conviction vs outcome (Spearman): {corr:+.2f} on {len(c)} buys — "
                  f"{'conviction is informative' if corr > 0.2 else 'conviction is not yet informative'}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')

    a = sub.add_parser('add', help='Log a decision')
    a.add_argument('--ticker', required=True)
    a.add_argument('--decision', required=True, choices=DECISIONS)
    a.add_argument('--thesis', required=True, help='One sentence, in your own words')
    a.add_argument('--mechanism', help='The concrete recovery mechanism (pillar 4)')
    a.add_argument('--conviction', type=int, default=3, choices=[1, 2, 3, 4, 5])
    a.add_argument('--pillars', help='Which pillars you judged met, e.g. "1,2,3,4,6"')
    a.add_argument('--size', type=float, help='Position size as %% of portfolio')
    a.add_argument('--horizon', type=int, default=180, help='Intended hold in days (default 180)')
    a.add_argument('--price', type=float, help='Override price (defaults to last close)')
    a.add_argument('--date', help='Override date (YYYY-MM-DD)')
    a.add_argument('--notes')

    l = sub.add_parser('list', help='Show logged decisions')
    l.add_argument('--decision', choices=DECISIONS)

    sub.add_parser('review', help='Score decisions against SPY')

    args = ap.parse_args()
    if args.cmd == 'add':
        cmd_add(args)
    elif args.cmd == 'list':
        cmd_list(args)
    elif args.cmd == 'review':
        cmd_review(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
