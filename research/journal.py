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

The file is append-only for anything that is a judgment. Change your mind by
logging a second entry, never by editing the first: the value of a row is that
it was written before the outcome was known. `edit` therefore refuses to touch
the decision, date, thesis, mechanism, conviction or pillars, and exists only to
correct facts that were already true - a fill price, a position size, a typo.

    venv/bin/python3 research/journal.py add --ticker BSX --decision buy \\
        --thesis "Restructuring plus Q2 beat; recall is contained and one-off" \\
        --mechanism "New management cost program" --conviction 4 --size 3
    venv/bin/python3 research/journal.py close --ticker BSX --price 61.40 \\
        --reason "Thesis played out; margin recovery is now consensus"
    venv/bin/python3 research/journal.py edit --ticker BSX --field price --value 49.55
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
           'thesis', 'pillars_met', 'size_pct', 'horizon_days', 'notes',
           'exit_date', 'exit_price', 'exit_reason']
DECISIONS = ('buy', 'watch', 'pass')

# What `edit` may touch. Facts can be wrong at entry - a fill price that differed
# from the close, a size not yet computed - and correcting them rewrites nothing.
FACT_FIELDS = ('price', 'size_pct', 'horizon_days', 'notes',
               'exit_date', 'exit_price', 'exit_reason')
# What it may not. Changing these after the fact turns the record into hindsight
# and makes `review` measure editing rather than judgment.
JUDGMENT_FIELDS = ('date', 'decision', 'thesis', 'mechanism', 'conviction', 'pillars_met')


def _dates_out(s):
    """Format a date column date-only, leaving blanks for missing values."""
    d = pd.to_datetime(s, format='mixed', errors='coerce')
    return d.dt.strftime('%Y-%m-%d').where(d.notna(), '')


def load():
    if not JOURNAL.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(JOURNAL)
    # 'mixed' tolerates files written before dates were normalised on save.
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    for c in COLUMNS:  # journals written before exit tracking existed
        if c not in df.columns:
            df[c] = ''
    # An all-empty column reads back as float64, and writing a date or reason into
    # it then warns about an incompatible dtype.
    for c in ('exit_date', 'exit_reason', 'notes', 'mechanism'):
        df[c] = df[c].astype(object)
    return df[COLUMNS]


def save(df):
    """Dates are written date-only so the file never gains mixed formats.

    Round-tripping a parsed Timestamp writes '2026-08-09 00:00:00' while a fresh
    entry writes '2026-08-09'. Once both are present, reading the file back
    raises, which surfaces as the journal breaking on the third entry.
    """
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out['date'] = _dates_out(out['date'])
    out['exit_date'] = _dates_out(out['exit_date'])
    out.to_csv(JOURNAL, index=False)


def _is_open(row):
    """A buy that has not been closed out."""
    return row['decision'] == 'buy' and not _has(row['exit_date'])


def _has(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() not in ('', 'nan', 'NaT')


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
        'exit_date': '',
        'exit_price': '',
        'exit_reason': '',
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
        if _has(r['exit_date']):
            ret = (float(r['exit_price']) / float(r['price']) - 1.0) * 100
            state = f"-> ${float(r['exit_price']):.2f} {str(r['exit_date'])[:10]} {ret:+.1f}%"
        else:
            state = ''
        print(f"  {r['date'].date()}  {r['decision']:5s}  {r['ticker']:6s} {px:>9s}  "
              f"conv {r['conviction']}  {str(r['thesis'])[:52]}")
        if state:
            print(f"  {'':43s}closed {state}")
    n_open = int(df.apply(_is_open, axis=1).sum())
    print(f"\n  {len(df)} decisions, {n_open} open position(s)")


def _forward(ticker, start, horizon_days, end=None):
    """Return (stock_pct, spy_pct) from `start` to `end`, else over the horizon.

    `end` is passed for closed positions so SPY is measured over the window the
    money was actually at risk, not a horizon that never ran its course.
    """
    try:
        import yfinance as yf
        start = pd.Timestamp(start)
        stop = (pd.Timestamp(end) if _has(end)
                else min(pd.Timestamp.now(), start + pd.Timedelta(days=horizon_days + 5)))
        data = yf.download([ticker, 'SPY'], start=start - pd.Timedelta(days=5),
                           end=stop + pd.Timedelta(days=1), progress=False, auto_adjust=True)
        px = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        px = px.dropna()
        if len(px) < 2:
            return np.nan, np.nan
        s = px[(px.index >= start) & (px.index <= stop)]
        if s.empty or len(s) < 2:
            return np.nan, np.nan
        r = (s.iloc[-1] / s.iloc[0] - 1.0) * 100
        return float(r.get(ticker, np.nan)), float(r.get('SPY', np.nan))
    except Exception:
        return np.nan, np.nan


def _select(df, ticker, date=None, decision=None, open_only=False):
    """Find the one row a command should act on, or explain the ambiguity."""
    t = ticker.upper()
    rows = df[df['ticker'] == t]
    if rows.empty:
        return None, f"no entries for {t}"
    if date:
        rows = rows[rows['date'] == pd.Timestamp(date)]
    if decision:
        rows = rows[rows['decision'] == decision]
    if open_only:
        rows = rows[rows.apply(_is_open, axis=1)]
    if rows.empty:
        return None, f"no matching entry for {t} (try journal.py list)"
    if len(rows) > 1:
        # Most recent is nearly always what was meant, but say so rather than guess silently.
        print(f"  {len(rows)} matching entries for {t}; using the most recent "
              f"({str(rows.iloc[-1]['date'])[:10]} {rows.iloc[-1]['decision']}). "
              f"Narrow with --date if that is wrong.")
    return rows.index[-1], None


def cmd_close(args):
    df = load()
    idx, err = _select(df, args.ticker, args.date_of, 'buy', open_only=True)
    if idx is None:
        print(f"  {err}. Only an open buy can be closed.")
        return
    row = df.loc[idx]
    exit_price = args.price if args.price else _price_now(row['ticker'])
    if not np.isfinite(exit_price):
        print("  could not determine an exit price; pass --price")
        return
    exit_date = args.date or datetime.now().strftime('%Y-%m-%d')
    df.at[idx, 'exit_date'] = exit_date
    df.at[idx, 'exit_price'] = exit_price
    df.at[idx, 'exit_reason'] = args.reason or ''
    save(df)

    entry = float(row['price'])
    ret = (exit_price / entry - 1.0) * 100
    _, spy = _forward(row['ticker'], row['date'], int(row['horizon_days'] or 180), end=exit_date)
    held = (pd.Timestamp(exit_date) - pd.Timestamp(row['date'])).days
    print(f"  closed {row['ticker']} @ ${exit_price:.2f} on {exit_date} "
          f"(entry ${entry:.2f}, held {held}d)")
    print(f"  realised {ret:+.1f}%", end='')
    if np.isfinite(spy):
        print(f"   SPY over the same window {spy:+.1f}%   excess {ret - spy:+.1f}pp")
    else:
        print()
    if args.reason:
        print(f"  reason: {args.reason}")


def cmd_edit(args):
    df = load()
    if args.field in JUDGMENT_FIELDS:
        print(f"  refusing to edit '{args.field}'.\n"
              f"  That is a judgment, and changing it after the fact is what makes a\n"
              f"  journal worthless - review would then be scoring hindsight. Log a new\n"
              f"  entry instead:  journal.py add --ticker {args.ticker.upper()} "
              f"--decision ... --thesis ...")
        return
    if args.field not in FACT_FIELDS:
        print(f"  unknown field '{args.field}'. Editable: {', '.join(FACT_FIELDS)}")
        return
    idx, err = _select(df, args.ticker, args.date_of, args.decision)
    if idx is None:
        print(f"  {err}")
        return
    value = args.value
    if args.field in ('price', 'size_pct', 'exit_price'):
        try:
            value = float(value)
        except ValueError:
            print(f"  '{value}' is not a number")
            return
    elif args.field == 'horizon_days':
        try:
            value = int(value)
        except ValueError:
            print(f"  '{value}' is not a whole number of days")
            return
    old = df.at[idx, args.field]
    df.at[idx, args.field] = value
    save(df)
    r = df.loc[idx]
    shown = '(empty)' if not _has(old) else old
    print(f"  {r['ticker']} {str(r['date'])[:10]} {r['decision']}: "
          f"{args.field} {shown} -> {value}")


def cmd_review(args):
    df = load()
    if df.empty:
        print("  no decisions logged yet")
        return
    print(f"  scoring {len(df)} decisions against SPY over the same windows...\n")
    rows = []
    for _, r in df.iterrows():
        closed = _has(r['exit_date']) and _has(r['exit_price'])
        end = r['exit_date'] if closed else None
        stock, spy = _forward(r['ticker'], r['date'], int(r['horizon_days'] or 180), end=end)
        if closed:
            # The realised fill is the truth; the price series only supplies SPY.
            stock = (float(r['exit_price']) / float(r['price']) - 1.0) * 100
        rows.append({**r.to_dict(), 'ret_pct': stock, 'spy_pct': spy,
                     'status': 'closed' if closed else 'open',
                     'excess_pp': stock - spy if np.isfinite(stock) and np.isfinite(spy) else np.nan})
    res = pd.DataFrame(rows).dropna(subset=['ret_pct'])
    if res.empty:
        print("  no resolvable outcomes yet")
        return

    print(f"  {'date':11s} {'dec':6s} {'ticker':7s} {'return':>8s} {'SPY':>8s} "
          f"{'excess':>9s} {'state':7s} thesis")
    for _, r in res.sort_values('date').iterrows():
        print(f"  {str(r['date'])[:10]:11s} {r['decision']:6s} {r['ticker']:7s} "
              f"{r['ret_pct']:+7.1f}% {r['spy_pct']:+7.1f}% {r['excess_pp']:+8.1f}pp "
              f"{r['status']:7s} {str(r['thesis'])[:38]}")

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

    c = sub.add_parser('close', help='Record a sale against an open buy')
    c.add_argument('--ticker', required=True)
    c.add_argument('--price', type=float, help='Fill price (defaults to last close)')
    c.add_argument('--date', help='Sale date (YYYY-MM-DD, defaults to today)')
    c.add_argument('--date-of', dest='date_of',
                   help='Entry date, if several buys of this ticker are open')
    c.add_argument('--reason', help='Why you sold - thesis played out, broke, better use of cash')

    e = sub.add_parser('edit', help='Correct a factual field on an existing entry')
    e.add_argument('--ticker', required=True)
    e.add_argument('--field', required=True,
                   help=f"One of: {', '.join(FACT_FIELDS)}")
    e.add_argument('--value', required=True)
    e.add_argument('--date-of', dest='date_of', help='Entry date, to disambiguate')
    e.add_argument('--decision', choices=DECISIONS, help='Decision type, to disambiguate')

    l = sub.add_parser('list', help='Show logged decisions')
    l.add_argument('--decision', choices=DECISIONS)

    sub.add_parser('review', help='Score decisions against SPY')

    args = ap.parse_args()
    if args.cmd == 'add':
        cmd_add(args)
    elif args.cmd == 'close':
        cmd_close(args)
    elif args.cmd == 'edit':
        cmd_edit(args)
    elif args.cmd == 'list':
        cmd_list(args)
    elif args.cmd == 'review':
        cmd_review(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
