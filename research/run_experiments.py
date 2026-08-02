"""Validation battery for the Laggard playbook.

Order matters here: establish a baseline, then test whether each pillar actually
earns its place (ablation), then choose the timing trigger and exit policy, then
check the result holds out of sample and outside technology.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import (DATA, DEFAULT_CFG, assign_industry, build_features, load,
                    make_signal, simulate, summarize)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)

CACHE = DATA / 'features.pkl'


def get_features(rebuild=False):
    if CACHE.exists() and not rebuild:
        with open(CACHE, 'rb') as fh:
            return pickle.load(fh)
    d = load()
    print('Assigning industries by trailing return correlation...')
    industry = assign_industry(d['close'], d['etf'])
    print('Building features...')
    f = build_features(d, industry)
    with open(CACHE, 'wb') as fh:
        pickle.dump((d, f), fh)
    return d, f


def run(d, f, label, **over):
    cfg = dict(DEFAULT_CFG)
    cfg.update(over)
    sig = make_signal(f, d, cfg)
    tr = simulate(sig, f, d, cfg)
    return tr, summarize(tr, label)


def show(rows, title):
    print()
    print('=' * 130)
    print(title)
    print('=' * 130)
    df = pd.DataFrame(rows)
    if df.empty or df['n'].sum() == 0:
        print('  no trades')
        return df
    print(df.round(1).to_string(index=False))
    return df


def main():
    d, f = get_features(rebuild='--rebuild' in sys.argv)
    close = d['close']
    print(f'Loaded {close.shape[0]} days x {close.shape[1]} tickers '
          f'({close.index[0].date()} .. {close.index[-1].date()})')

    # ---------------------------------------------------------------- 1. baselines
    rows = []

    # Buy-and-hold benchmark for context: every stock, every week, 6m hold.
    spy = d['etf']['SPY']
    fwd = close.shift(-126) / close.shift(-1) - 1.0
    spy_fwd = spy.shift(-126) / spy.shift(-1) - 1.0
    member = d['member'].reindex(columns=close.columns).fillna(False)
    weekly = close.index[close.index.dayofweek == 4]
    universe_fwd = fwd.loc[weekly][member.loc[weekly]]
    uni = universe_fwd.stack().dropna() * 100
    uni_ex = (fwd.loc[weekly][member.loc[weekly]].sub(spy_fwd.loc[weekly], axis=0)
              ).stack().dropna() * 100
    rows.append({
        'label': 'ALL index members (random 6m hold)',
        'n': len(uni), 'n_names': close.shape[1],
        'win_%': (uni > 0).mean() * 100, 'mean_%': uni.mean(), 'median_%': uni.median(),
        'beat_spy_%': (uni_ex > 0).mean() * 100,
        'mean_excess_%': uni_ex.mean(), 'median_excess_%': uni_ex.median(),
        'p10_%': uni.quantile(.1), 'p90_%': uni.quantile(.9),
        'mean_mae_%': np.nan, 'worst_%': uni.min(), 'avg_hold': 126,
    })

    tr_base, s = run(d, f, 'FULL playbook (5 pillars)')
    rows.append(s)

    show(rows, '1. BASELINE  --  does the playbook beat simply owning index members?')

    # ------------------------------------------------------------------ 2. ablation
    rows = []
    _, s = run(d, f, 'Full playbook'); rows.append(s)
    _, s = run(d, f, '- drop DAMAGE (no drawdown req)', max_dd=0.0, min_peak_age=0); rows.append(s)
    _, s = run(d, f, '- drop INDUSTRY tailwind', require_industry_up=False); rows.append(s)
    _, s = run(d, f, '- drop DIVERGENCE', require_divergence=False); rows.append(s)
    _, s = run(d, f, '- drop TIMING trigger', trigger='none'); rows.append(s)
    _, s = run(d, f, 'ONLY damage + timing', require_industry_up=False,
               require_divergence=False); rows.append(s)
    _, s = run(d, f, 'ONLY damage (no industry/div/timing)', require_industry_up=False,
               require_divergence=False, trigger='none'); rows.append(s)
    show(rows, '2. ABLATION  --  which pillars actually earn their keep?')

    # ------------------------------------------------------------------- 3. trigger
    rows = []
    for trig in ('none', 'above_200', 'reclaim_200', 'reclaim_50', 'breakout_3m',
                 'higher_low'):
        _, s = run(d, f, f'trigger = {trig}', trigger=trig)
        rows.append(s)
    show(rows, '3. TIMING TRIGGER  --  how do you know the decline stopped?')

    # ---------------------------------------------------------------------- 4. exit
    rows = []
    for hold in (21, 63, 126, 189, 252):
        _, s = run(d, f, f'time stop {hold}d', max_hold=hold)
        rows.append(s)
    for stop, tgt in ((0.15, None), (0.20, None), (0.25, None), (0.20, 0.50)):
        _, s = run(d, f, f'stop {stop:.0%} target {tgt}', max_hold=252,
                   stop_pct=stop, target_pct=tgt)
        rows.append(s)
    for trail in (0.15, 0.20, 0.25):
        _, s = run(d, f, f'trailing {trail:.0%} (max 252d)', max_hold=252, trail_pct=trail)
        rows.append(s)
    _, s = run(d, f, 'thesis break: 5d below 200dma', max_hold=252, thesis_break_days=5)
    rows.append(s)
    _, s = run(d, f, 'thesis break + 20% stop', max_hold=252, thesis_break_days=5,
               stop_pct=0.20)
    rows.append(s)
    show(rows, '4. EXIT POLICY  --  multi-month holds need a defined way out')

    # ------------------------------------------------------------------- 5. regimes
    tr = tr_base.copy()
    tr['year'] = tr['entry_date'].dt.year
    print()
    print('=' * 130)
    print('5. BY ENTRY YEAR  --  is this just the 2023-2026 tech bull market?')
    print('=' * 130)
    by_year = tr.groupby('year').agg(
        n=('ret', 'size'), win_pct=('ret', lambda s: (s > 0).mean() * 100),
        median_pct=('ret', 'median'), mean_pct=('ret', 'mean'),
        median_excess=('excess', 'median'), beat_spy_pct=('excess', lambda s: (s > 0).mean() * 100),
        mean_mae=('mae', 'mean'))
    print(by_year.round(1).to_string())

    print()
    print('=' * 130)
    print('6. BY INDUSTRY  --  does it generalize beyond semis/tech?')
    print('=' * 130)
    by_ind = tr.groupby('industry').agg(
        n=('ret', 'size'), win_pct=('ret', lambda s: (s > 0).mean() * 100),
        median_pct=('ret', 'median'), median_excess=('excess', 'median'),
        beat_spy_pct=('excess', lambda s: (s > 0).mean() * 100))
    print(by_ind.sort_values('n', ascending=False).round(1).to_string())

    # ------------------------------------------------------- 7. train / holdout split
    print()
    print('=' * 130)
    print('7. OUT-OF-SAMPLE  --  tuned on 2007-2018, verified on 2019-2026')
    print('=' * 130)
    tr_in = tr[tr['entry_date'] < '2019-01-01']
    tr_out = tr[tr['entry_date'] >= '2019-01-01']
    print(pd.DataFrame([summarize(tr_in, 'IN-SAMPLE 2007-2018'),
                        summarize(tr_out, 'HOLDOUT 2019-2026')]).round(1).to_string(index=False))

    tr_base.to_csv(DATA / 'trades_baseline.csv', index=False)
    print(f'\nSaved {len(tr_base)} baseline trades -> research/data/trades_baseline.csv')

    print()
    print('Top 15 winners:')
    print(tr_base.nlargest(15, 'ret')[
        ['entry_date', 'ticker', 'industry', 'dd_3y', 'rel_12m', 'ret', 'excess', 'reason']
    ].round(1).to_string(index=False))
    print()
    print('Worst 15:')
    print(tr_base.nsmallest(15, 'ret')[
        ['entry_date', 'ticker', 'industry', 'dd_3y', 'rel_12m', 'ret', 'excess', 'mae', 'reason']
    ].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
