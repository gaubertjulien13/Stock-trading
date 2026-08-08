"""The question the earlier work never asked.

Everything so far tested SELECTION: can a rule pick names better than the index?
Answer: no. But that is not how the Intel trade was actually made. The name came
from reading and conviction, not from a screen. What a script could still add is
EXECUTION: given a name you have already decided you want, when do you buy it,
how much, and when do you get out?

That is a conditional question and it has a different answer. Here the candidate
set is held fixed -- names shaped like Intel in mid-2025 -- and only the entry
timing and exit policy are varied.

Run:  venv/bin/python3 research/validate_timing.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 60)

MAX_WAIT = 252          # give a trigger one year to appear, then abandon the idea


def onsets(cond, cooldown=252):
    """First day the setup turns true, debounced so one idea = one observation."""
    v = cond.values
    out = np.zeros_like(v, dtype=bool)
    last = np.full(v.shape[1], -10 ** 9)
    for i in range(v.shape[0]):
        hit = np.where(v[i])[0]
        for j in hit:
            if i - last[j] >= cooldown:
                out[i, j] = True
                last[j] = i
    return pd.DataFrame(out, index=cond.index, columns=cond.columns)


def run_policies(d, f):
    close, open_ = d['close'], d['open']
    spy = d['etf']['SPY'].reindex(close.index).ffill()
    base = build(d, f)

    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_up = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    cond = base & damaged & ind_up
    ons = onsets(cond)
    print(f'Intel-shaped ideas in the sample: {int(ons.values.sum())}')

    C = close.values
    O = open_.reindex(index=close.index, columns=close.columns).values
    L = d['low'].reindex(index=close.index, columns=close.columns).values
    H = d['high'].reindex(index=close.index, columns=close.columns).values
    S = spy.values
    SMA200 = f['sma200'].values
    above200 = f['above_200'].values
    above50 = f['above_50'].values
    lo3 = close.rolling(63, min_periods=40).min().values
    lo6 = close.rolling(126, min_periods=80).min().values
    n = len(close.index)

    triggers = {
        'buy immediately': None,
        'wait: reclaim 200dma': lambda k, j: bool(above200[k, j]),
        'wait: higher low + above 50dma': lambda k, j: bool(
            np.isfinite(lo3[k, j]) and np.isfinite(lo6[k, j])
            and lo3[k, j] > lo6[k, j] * 1.02 and above50[k, j]),
    }

    rows = []
    entries = {}
    for tname, fn in triggers.items():
        recs = []
        for i, j in zip(*np.where(ons.values)):
            if fn is None:
                k = i
            else:
                k = None
                for kk in range(i, min(i + MAX_WAIT, n - 1)):
                    if fn(kk, j):
                        k = kk
                        break
                if k is None:
                    recs.append({'traded': False})
                    continue
            e = k + 1
            if e + 252 >= n:
                continue
            ep = O[e, j]
            if not np.isfinite(ep) or ep <= 0:
                continue
            rec = {'traded': True, 'i': i, 'entry_i': e, 'j': j, 'entry': ep,
                   'wait_days': e - i}
            for h, lbl in ((126, '6m'), (252, '12m')):
                x = e + h
                if x >= n or not np.isfinite(C[x, j]):
                    continue
                rec[f'ret_{lbl}'] = (C[x, j] / ep - 1) * 100
                rec[f'ex_{lbl}'] = rec[f'ret_{lbl}'] - (S[x] / S[e] - 1) * 100
            seg = L[e:min(e + 252, n), j]
            rec['mae_12m'] = (np.nanmin(seg) / ep - 1) * 100 if np.isfinite(seg).any() else np.nan
            recs.append(rec)
        df = pd.DataFrame(recs)
        entries[tname] = df[df['traded']].copy() if 'traded' in df else df
        t = entries[tname]
        rows.append({
            'entry policy': tname,
            'ideas acted on %': 100 * len(t) / max(len(df), 1),
            'n': len(t),
            'median wait (days)': t['wait_days'].median() if len(t) else np.nan,
            'median 6m %': t['ret_6m'].median(),
            'median 6m vs SPY': t['ex_6m'].median(),
            'median 12m %': t['ret_12m'].median(),
            'median 12m vs SPY': t['ex_12m'].median(),
            'beat SPY 12m %': (t['ex_12m'] > 0).mean() * 100,
            'median worst dip %': t['mae_12m'].median(),
            'p10 worst dip %': t['mae_12m'].quantile(0.10),
        })

    print()
    print('=' * 150)
    print('TEST A  --  same ideas, different entry timing. Forward returns from entry.')
    print('=' * 150)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    return entries, (C, O, L, H, S, SMA200), close


def exit_policies(entries, arrays, close):
    """On the reclaim entries only: which exit rule keeps the most of the move?"""
    C, O, L, H, S, SMA200 = arrays
    n = len(close.index)
    t = entries['wait: reclaim 200dma']

    policies = {
        'hold 6 months': dict(max_hold=126),
        'hold 12 months': dict(max_hold=252),
        'hold 12m, hard stop -20%': dict(max_hold=252, stop=0.20),
        'hold 12m, trailing stop -20%': dict(max_hold=252, trail=0.20),
        'hold 12m, trailing stop -25%': dict(max_hold=252, trail=0.25),
        'hold 12m, exit on losing 200dma 10d': dict(max_hold=252, thesis=10),
        'hold 24 months': dict(max_hold=504),
        'hold 24m, trailing stop -25%': dict(max_hold=504, trail=0.25),
    }

    rows = []
    for label, p in policies.items():
        max_hold = p['max_hold']
        stop, trail, thesis = p.get('stop'), p.get('trail'), p.get('thesis')
        res = []
        for _, r in t.iterrows():
            e, j, ep = int(r['entry_i']), int(r['j']), r['entry']
            end = min(e + max_hold, n - 1)
            if end <= e:
                continue
            exit_i, exit_p, reason = end, C[end, j], 'time'
            peak, below = ep, 0
            for k in range(e, end + 1):
                cl = C[k, j]
                if not np.isfinite(cl):
                    continue
                if stop is not None and np.isfinite(L[k, j]) and L[k, j] <= ep * (1 - stop):
                    exit_i, exit_p, reason = k, ep * (1 - stop), 'stop'
                    break
                peak = max(peak, H[k, j] if np.isfinite(H[k, j]) else cl)
                if trail is not None and cl <= peak * (1 - trail):
                    exit_i, exit_p, reason = k, cl, 'trail'
                    break
                if thesis is not None and np.isfinite(SMA200[k, j]):
                    below = below + 1 if cl < SMA200[k, j] else 0
                    if below >= thesis:
                        exit_i, exit_p, reason = k, cl, 'thesis'
                        break
            if not np.isfinite(exit_p):
                continue
            ret = (exit_p / ep - 1) * 100
            bench = (S[exit_i] / S[e] - 1) * 100
            res.append({'ret': ret, 'ex': ret - bench, 'hold': exit_i - e, 'reason': reason})
        r = pd.DataFrame(res)
        if r.empty:
            continue
        rows.append({
            'exit policy': label, 'n': len(r),
            'median %': r['ret'].median(), 'mean %': r['ret'].mean(),
            'median vs SPY': r['ex'].median(), 'mean vs SPY': r['ex'].mean(),
            'win %': (r['ret'] > 0).mean() * 100,
            'beat SPY %': (r['ex'] > 0).mean() * 100,
            'p10 %': r['ret'].quantile(0.10),
            'p90 %': r['ret'].quantile(0.90),
            'avg hold (d)': r['hold'].mean(),
        })

    print()
    print('=' * 150)
    print('TEST B  --  same entries (200dma reclaim), different exit policy.')
    print('=' * 150)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    return pd.DataFrame(rows)


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    entries, arrays, close = run_policies(d, f)
    ex = exit_policies(entries, arrays, close)

    print()
    print('=' * 150)
    print('TEST C  --  how long does the wait cost you, and what does it save you?')
    print('=' * 150)
    imm = entries['buy immediately']
    rec = entries['wait: reclaim 200dma']
    print(f"Buying immediately: median worst dip over the next 12 months "
          f"{imm['mae_12m'].median():.1f}%, and 1 idea in 10 fell "
          f"{imm['mae_12m'].quantile(0.10):.1f}% or worse.")
    print(f"Waiting for the reclaim: median worst dip {rec['mae_12m'].median():.1f}%, "
          f"1 in 10 fell {rec['mae_12m'].quantile(0.10):.1f}% or worse.")
    print(f"Median wait: {rec['wait_days'].median():.0f} trading days "
          f"({rec['wait_days'].median() / 21:.1f} months).")

    ex.to_csv(DATA / 'verify_exits.csv', index=False)
    for k, v in entries.items():
        v.to_csv(DATA / f"verify_entry_{k.split(':')[0].replace(' ', '_')}.csv", index=False)
    print('\nSaved -> research/data/verify_exits.csv')


if __name__ == '__main__':
    main()
