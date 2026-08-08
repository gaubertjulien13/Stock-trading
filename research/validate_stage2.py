"""Does the Stage-2 score rank forward returns? The one open question in FUNNEL.md.

PLAYBOOK.md already established that the Stage-1 filter has no edge against SPY.
That leaves a narrower and still-unanswered question: *within* the Stage-1 pool,
does a higher pillar score mean a better stock? If it does, the funnel earns its
keep as a ranker even though the filter does not. If it does not, the score is
decoration and the ranking should not influence position sizing.

Method mirrors PLAYBOOK conventions:
  - Point-in-time membership and features, monthly as-of dates, 2007 onward.
  - Benchmark is the Stage-1 pool itself (what picking at random from the same
    candidates would have given), not zero and not SPY - that isolates the
    score's contribution from the filter's.
  - Every headline number is paired with a random-draw null of the same size,
    because the whole lesson of this repo is that small books are mostly luck.

Pillars 4 and 5 need headlines, which do not exist historically. They are held at
their production placeholders (5.0 / 3.0), exactly as recommend.py behaves without
--news. This therefore tests the *mechanical* score - the part that runs unattended.

    venv/bin/python3 research/validate_stage2.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from recommend import stage2

pd.set_option('display.width', 220)

HORIZONS = {'3m': 63, '6m': 126, '12m': 252}
START = '2007-01-01'
N_NULL = 500


def stage1_matrix(f, d):
    """Vectorized twin of recommend.stage1_mask - identical rules, computed once."""
    close = d['close']
    member = d['member'].reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)
    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_ok = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    diverge = f['rel_12m'] <= -25.0
    seasoned = f['age_days'] >= 400
    return member & liquid & damaged & ind_ok & diverge & seasoned & close.notna()


def month_ends(index, start):
    idx = index[index >= pd.Timestamp(start)]
    return [g.max() for _, g in pd.Series(idx, index=idx).groupby(idx.to_period('M'))]


def collect(d, f, mask, dates):
    close = d['close']
    fwd = {k: close.shift(-h) / close - 1.0 for k, h in HORIZONS.items()}
    spy = d['etf']['SPY']
    spy_fwd = {k: spy.shift(-h) / spy - 1.0 for k, h in HORIZONS.items()}
    member = d['member'].reindex(columns=close.columns).fillna(False)

    rows = []
    for i, asof in enumerate(dates, 1):
        row = mask.loc[asof]
        cands = row[row].index.tolist()
        if len(cands) < 5:
            continue
        print(f"\r  scoring {i}/{len(dates)}  {asof.date()}  ({len(cands)} candidates)   ",
              end='', flush=True)
        scored = stage2(cands, asof, f, d, buys=None, fetch_news=False)
        if scored.empty:
            continue
        mem = member.loc[asof]
        mem_cols = mem[mem].index
        for _, s in scored.iterrows():
            r = {'asof': asof, 'ticker': s['ticker'], 'score': s['score'],
                 'band': s['band'], 'tag': s['tag'], 'dd': s['dd_3y_pct']}
            for k in HORIZONS:
                r[f'fwd_{k}'] = fwd[k].at[asof, s['ticker']]
                r[f'spy_{k}'] = spy_fwd[k].get(asof, np.nan)
                r[f'univ_{k}'] = fwd[k].loc[asof, mem_cols].mean()
            rows.append(r)
    print("\r" + " " * 70 + "\r", end='')
    return pd.DataFrame(rows)


def rank_ic(df, h):
    """Spearman correlation of score vs forward return, computed per date.

    Per-date is the honest way: pooling all dates would let a few good months
    masquerade as cross-sectional skill.
    """
    ics = []
    for asof, g in df.groupby('asof'):
        g = g.dropna(subset=[f'fwd_{h}', 'score'])
        if len(g) < 8 or g['score'].nunique() < 3:
            continue
        ics.append(g['score'].corr(g[f'fwd_{h}'], method='spearman'))
    if not ics:
        return np.nan, np.nan, 0
    ics = np.array(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else np.nan
    return ics.mean(), t, len(ics)


def null_draw(df, h, size_by_date, rng, runs=N_NULL):
    """Mean forward return of same-size random draws from the same dates' pools."""
    out = []
    groups = {a: g[f'fwd_{h}'].dropna().to_numpy() for a, g in df.groupby('asof')}
    for _ in range(runs):
        vals = []
        for asof, k in size_by_date.items():
            pool = groups.get(asof)
            if pool is None or len(pool) == 0 or k == 0:
                continue
            vals.append(rng.choice(pool, size=min(k, len(pool)), replace=False).mean())
        if vals:
            out.append(float(np.mean(vals)))
    return np.array(out)


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    print(f"loaded panel: {d['close'].shape[0]} days x {d['close'].shape[1]} tickers")

    mask = stage1_matrix(f, d)
    dates = month_ends(d['close'].index, START)
    print(f"as-of dates: {len(dates)}  ({dates[0].date()} -> {dates[-1].date()})")

    df = collect(d, f, mask, dates)
    print(f"\ncandidate-months scored: {len(df)}   unique names: {df['ticker'].nunique()}")
    print(f"band mix: " + "  ".join(f"{b}={n}" for b, n in df['band'].value_counts().items()))

    print(f"\n{'=' * 88}")
    print("  1. DOES THE SCORE RANK FORWARD RETURNS?  (Spearman IC per date)")
    print(f"{'=' * 88}")
    print(f"  {'horizon':8s} {'mean IC':>9s} {'t-stat':>8s} {'months':>8s}   interpretation")
    for h in HORIZONS:
        ic, t, n = rank_ic(df, h)
        verdict = ('noise' if abs(ic) < 0.03 else
                   'weak' if abs(ic) < 0.06 else 'notable')
        print(f"  {h:8s} {ic:+9.4f} {t:+8.2f} {n:8d}   {verdict}")
    print("  (|IC| < 0.03 is indistinguishable from noise; PLAYBOOK found 0.01-0.02 "
          "for the\n   dd_3y and rel_12m factors that constitute the Intel thesis.)")

    print(f"\n{'=' * 88}")
    print("  2. BAND PERFORMANCE vs THE POOL IT WAS SELECTED FROM")
    print(f"{'=' * 88}")
    for h in HORIZONS:
        col, uc, sc = f'fwd_{h}', f'univ_{h}', f'spy_{h}'
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        pool_mean = sub[col].mean()
        print(f"\n  --- {h} forward ---   Stage-1 pool mean {pool_mean * 100:+.2f}%   "
              f"S&P members {sub[uc].mean() * 100:+.2f}%   SPY {sub[sc].mean() * 100:+.2f}%")
        print(f"  {'band':8s} {'n':>6s} {'mean':>8s} {'median':>8s} {'vs pool':>9s} "
              f"{'vs SPY':>9s} {'beat SPY':>9s}")
        for b in ('STRONG', 'WATCH', 'WEAK'):
            g = sub[sub['band'] == b]
            if g.empty:
                continue
            print(f"  {b:8s} {len(g):6d} {g[col].mean() * 100:+7.2f}% "
                  f"{g[col].median() * 100:+7.2f}% {(g[col].mean() - pool_mean) * 100:+8.2f}pp "
                  f"{(g[col].mean() - g[sc].mean()) * 100:+8.2f}pp "
                  f"{(g[col] > g[sc]).mean() * 100:8.0f}%")

    print(f"\n{'=' * 88}")
    print("  3. IS 'STRONG' DISTINGUISHABLE FROM A RANDOM DRAW OF THE SAME SIZE?")
    print(f"{'=' * 88}")
    rng = np.random.default_rng(7)
    for h in HORIZONS:
        sub = df.dropna(subset=[f'fwd_{h}'])
        strong = sub[sub['band'] == 'STRONG']
        if strong.empty:
            print(f"  {h}: no STRONG observations")
            continue
        size_by_date = strong.groupby('asof').size().to_dict()
        null = null_draw(sub, h, size_by_date, rng)
        actual = strong.groupby('asof')[f'fwd_{h}'].mean().mean()
        pct = float((null < actual).mean() * 100)
        print(f"  {h:4s}  STRONG {actual * 100:+6.2f}%   random draws "
              f"{null.mean() * 100:+6.2f}% (5th-95th: {np.percentile(null, 5) * 100:+.2f}% "
              f"to {np.percentile(null, 95) * 100:+.2f}%)   percentile {pct:.0f}")
    print("\n  A percentile below ~95 means STRONG is inside the range of random selection\n"
          "  from the same candidate pool - i.e. the banding added nothing detectable.")

    print(f"\n{'=' * 88}")
    print("  4. SCORE QUINTILES (monotonic ranking would show a clean gradient)")
    print(f"{'=' * 88}")
    for h in HORIZONS:
        sub = df.dropna(subset=[f'fwd_{h}']).copy()
        if sub['score'].nunique() < 5:
            continue
        sub['q'] = pd.qcut(sub['score'], 5, labels=['Q1 low', 'Q2', 'Q3', 'Q4', 'Q5 high'],
                           duplicates='drop')
        g = sub.groupby('q', observed=True)[f'fwd_{h}'].agg(['count', 'mean', 'median'])
        line = "  ".join(f"{q}: {r['mean'] * 100:+.2f}%" for q, r in g.iterrows())
        spread = (g['mean'].iloc[-1] - g['mean'].iloc[0]) * 100
        print(f"  {h:4s}  {line}    Q5-Q1 spread {spread:+.2f}pp")

    out = DATA / 'stage2_validation.csv'
    df.to_csv(out, index=False)
    print(f"\n  detail written to {out}")


if __name__ == '__main__':
    main()
