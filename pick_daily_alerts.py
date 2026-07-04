"""
Diversification-aware daily pick helper.

Reads alert_log.csv and ranks the day's non-vetoed alerts for a small account
(1-5 positions). The ranking encodes what the multi-regime backtest actually
validated — and deliberately NOT what intuition suggests:

  1. SECTOR DIVERSITY first: taking the day's top scores concentrates into
     correlated, extended names and halved expectancy in the backtest. One
     pick per sector.
  2. MILD DIP preferred: entries whose own 5-day move was -6%..0% carried the
     bulk of the edge (+0.27..0.30R vs +0.10 elsewhere).
  3. VOLUME SURGE next: the one score component that discriminated (+0.21R
     with vs +0.17R without).
  4. SCORE is only a tiebreak: score>=14 is fine, but chasing the max score
     of the day was proven counterproductive.

Usage:
    python pick_daily_alerts.py                # today's alerts, top 5
    python pick_daily_alerts.py --date 2026-07-02 --max-picks 3
    python pick_daily_alerts.py --log alert_log.csv
"""

import argparse
from datetime import datetime

import pandas as pd


def rank_alerts(df, max_picks=5):
    """Return (picks, alternates) DataFrames from one day's alerts."""
    df = df.copy()

    # Latest alert per ticker (a ticker can re-alert after the debounce)
    df['ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('ts').groupby('ticker', as_index=False).last()

    df['stk_ret_5d'] = pd.to_numeric(df.get('stk_ret_5d'), errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    df['vol_surge'] = df.get('volume_surge').astype(str).str.lower().eq('true')

    df['mild_dip'] = df['stk_ret_5d'].between(-6.0, 0.0, inclusive='neither')
    df['rank_key'] = (
        df['mild_dip'].astype(int) * 100        # dominant: buy mild dips
        + df['vol_surge'].astype(int) * 10      # then: volume participation
        + df['score']                            # tiebreak only
    )
    df = df.sort_values('rank_key', ascending=False)

    picks, seen_sectors = [], set()
    for _, row in df.iterrows():
        sector = str(row.get('sector', 'Unknown')).strip() or 'Unknown'
        if sector in seen_sectors:
            continue
        seen_sectors.add(sector)
        picks.append(row)
        if len(picks) >= max_picks:
            break

    picked = {r['ticker'] for r in picks}
    alternates = df[~df['ticker'].isin(picked)]
    return pd.DataFrame(picks), alternates


def main():
    parser = argparse.ArgumentParser(description="Rank today's alerts for a few-position account")
    parser.add_argument("--log", default="alert_log.csv", help="Path to alert log CSV")
    parser.add_argument("--date", default=None, help="Date to analyze, YYYY-MM-DD (default: today)")
    parser.add_argument("--max-picks", type=int, default=5, help="Max picks (default 5)")
    parser.add_argument("--min-score", type=int, default=11, help="Ignore alerts below this score")
    args = parser.parse_args()

    day = args.date or datetime.today().strftime('%Y-%m-%d')
    df = pd.read_csv(args.log)
    df = df[df['timestamp'].str.startswith(day)]
    df = df[df['veto_reason'].fillna('') == '']
    df = df[pd.to_numeric(df['score'], errors='coerce') >= args.min_score]

    if df.empty:
        print(f"No non-vetoed alerts >= {args.min_score} pts on {day}.")
        return

    picks, alternates = rank_alerts(df, max_picks=args.max_picks)

    print(f"\n=== Picks for {day} (max {args.max_picks}, one per sector) ===\n")
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        why = []
        if r['mild_dip']:
            why.append(f"mild dip {r['stk_ret_5d']:+.1f}%")
        elif pd.notna(r['stk_ret_5d']):
            why.append(f"5d {r['stk_ret_5d']:+.1f}%")
        if r['vol_surge']:
            why.append("volume surge")
        why.append(f"score {r['score']:.0f}")
        print(f"  {i}. {r['ticker']:6s} ${float(r['price']):>8.2f}  "
              f"{str(r.get('sector','?'))[:22]:22s}  [{', '.join(why)}]")
        print(f"     stop ${float(r['stop_1x']):.2f} | target ${float(r['target_3x']):.2f}")

    if len(alternates):
        alt = ", ".join(f"{r.ticker}({r.score:.0f})" for r in alternates.head(10).itertuples())
        print(f"\n  Alternates (same-sector or lower-ranked): {alt}")
    print()


if __name__ == "__main__":
    main()
