"""One-off: simulate this week's logged alerts (Jul 6-10) against 15m market data.

Entry = logged alert price. Exit = first touch of stop_1x or target_3x
(stop wins ties, conservative). Unresolved trades are marked to Friday's close.
"""
import numpy as np
import pandas as pd
import yfinance as yf

WEEK_START, WEEK_END = '2026-07-06', '2026-07-11'

df = pd.read_csv('alert_log.csv')
wk = df[(df['timestamp'] >= WEEK_START) & (df['timestamp'] < WEEK_END)].copy()
wk['ts'] = (pd.to_datetime(wk['timestamp'])
            .dt.tz_localize('America/Los_Angeles')
            .dt.tz_convert('America/New_York'))
wk['day'] = wk['timestamp'].str[:10]
wk['vetoed'] = wk['veto_reason'].fillna('') != ''
for c in ('price', 'stop_1x', 'target_3x', 'score', 'stk_ret_5d'):
    wk[c] = pd.to_numeric(wk[c], errors='coerce')
wk = wk.dropna(subset=['price', 'stop_1x', 'target_3x'])

tickers = sorted(wk['ticker'].unique())
print(f"downloading 15m bars for {len(tickers)} tickers...")
bars = {}
for i in range(0, len(tickers), 60):
    batch = tickers[i:i + 60]
    data = yf.download(batch, start=WEEK_START, end='2026-07-12', interval='15m',
                       progress=False, group_by='ticker', threads=True)
    for t in batch:
        try:
            sub = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
            if len(sub):
                bars[t] = sub
        except Exception:
            pass
print(f"got data for {len(bars)} tickers")


def simulate(row):
    b = bars.get(row['ticker'])
    if b is None:
        return None
    fut = b[b.index > row['ts']]
    if fut.empty:
        return None
    entry, stop, target = row['price'], row['stop_1x'], row['target_3x']
    risk = entry - stop
    if risk <= 0:
        return None
    for _, bar in fut.iterrows():
        if bar['Low'] <= stop:
            return {'outcome': 'stop', 'r': -1.0, 'ret': (stop / entry - 1) * 100}
        if bar['High'] >= target:
            return {'outcome': 'target', 'r': (target - entry) / risk,
                    'ret': (target / entry - 1) * 100}
    last = fut['Close'].iloc[-1]
    return {'outcome': 'open', 'r': (last - entry) / risk, 'ret': (last / entry - 1) * 100}


def cohort_stats(rows, name):
    res = [dict(simulate(r), ticker=r['ticker'], day=r['day'], score=r['score'])
           for _, r in rows.iterrows() if simulate(r)]
    if not res:
        print(f"{name:42s}  no trades")
        return None
    c = pd.DataFrame(res)
    n = len(c)
    print(f"{name:42s} n={n:4d}  avgR={c['r'].mean():+.3f}  avg%={c['ret'].mean():+.2f}%  "
          f"tgt={((c['outcome'] == 'target').sum() / n * 100):4.1f}%  "
          f"stop={((c['outcome'] == 'stop').sum() / n * 100):4.1f}%  "
          f"win={(c['ret'] > 0).mean() * 100:4.1f}%")
    return c

# --- cohorts ---
nv = wk[~wk['vetoed']]
first_week = nv.sort_values('timestamp').groupby('ticker').head(1)      # 1 entry/ticker/week
first_day = nv.sort_values('timestamp').groupby(['ticker', 'day']).head(1)

# reconstructed daily picks: non-vetoed rows logged by 07:31 PT, ranked as the picks email does
import sys
sys.path.insert(0, '.')
from pick_daily_alerts import rank_alerts
picks_rows = []
for day, grp in nv.groupby('day'):
    early = grp[grp['timestamp'].str[11:16] <= '07:31']
    if early.empty:
        continue
    p, _ = rank_alerts(early, max_picks=3)
    for _, r in p.iterrows():
        src = grp[(grp['ticker'] == r['ticker'])].sort_values('timestamp').iloc[0]
        picks_rows.append(src)
picks = pd.DataFrame(picks_rows)

vetoed_1 = wk[wk['vetoed']].sort_values('timestamp').groupby('ticker').head(1)

print(f"\n{'=' * 105}")
print("  WEEK Jul 6-10 PERFORMANCE  (entry=alert price, stop 1xATR, target 3xATR, else mark at Fri close)")
print(f"{'=' * 105}")
cohort_stats(first_week, "ALL non-vetoed (1st alert/ticker/week)")
cohort_stats(first_day, "ALL non-vetoed (1st alert/ticker/day)")
c_picks = cohort_stats(picks, "DAILY PICKS top-3 (reconstructed)")
cohort_stats(first_week[first_week['score'] >= 14], "score >= 14 cohort")
mild = first_week[(first_week['stk_ret_5d'] > -6) & (first_week['stk_ret_5d'] < 0)]
cohort_stats(mild, "mild-dip cohort (email tier)")
cohort_stats(vetoed_1, "VETOED (counterfactual, 1st/ticker)")

if c_picks is not None and len(c_picks):
    print("\n  Daily picks detail:")
    for _, r in c_picks.sort_values('day').iterrows():
        print(f"    {r['day']}  {r['ticker']:6s} score={r['score']:.0f}  "
              f"{r['outcome']:6s}  R={r['r']:+.2f}  ret={r['ret']:+.2f}%")

spy = yf.download('SPY', start=WEEK_START, end='2026-07-12', interval='1d', progress=False)
spy_ret = (spy['Close'].iloc[-1] / spy['Open'].iloc[0] - 1) * 100
print(f"\n  SPY over the same week: {float(spy_ret):+.2f}%")
