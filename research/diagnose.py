"""Find corrupted price series and understand why INTC worked when the screen didn't."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA, DEFAULT_CFG, make_signal, simulate

pd.set_option('display.width', 250)

with open(DATA / 'features.pkl', 'rb') as fh:
    d, f = pickle.load(fh)

close = d['close']

print('=' * 90)
print('DATA QUALITY: implausible single-day moves (split/adjustment artifacts)')
print('=' * 90)
rets = close.pct_change(fill_method=None)
bad = (rets.abs() > 1.5).sum().sort_values(ascending=False)
bad = bad[bad > 0]
print(f'Tickers with any single-day move >150%: {len(bad)}')
print(bad.head(25).to_string())

print()
print('Largest single-day moves in the dataset:')
stacked = rets.stack().dropna()
top = stacked.abs().nlargest(15)
for (dt, tk), _ in top.items():
    print(f'  {dt.date()} {tk:<6} {rets.loc[dt, tk] * 100:>10.0f}%  '
          f'price {close.loc[:dt, tk].iloc[-2]:.2f} -> {close.loc[dt, tk]:.2f}')

print()
print('=' * 90)
print('WHY INTC WORKED: feature values at the winning entry vs. the screen')
print('=' * 90)
for date in ['2025-08-12', '2025-09-15', '2024-10-31', '2023-05-26']:
    ts = pd.Timestamp(date)
    if ts not in close.index:
        ts = close.index[close.index.get_indexer([ts], method='nearest')[0]]
    print(f'\nINTC on {ts.date()}:')
    for k in ('dd_3y', 'peak_age_days', 'ret_12m', 'ret_6m', 'ret_3m', 'ret_1m',
              'rel_12m', 'rel_3m', 'ind_ret_12m', 'dist_200_pct', 'sma200_slope',
              'off_52w_low_pct', 'vol_ann'):
        try:
            v = f[k].loc[ts, 'INTC']
            print(f'   {k:<18} {v:>10.1f}')
        except Exception:
            pass
    print(f'   {"industry":<18} {f["industry"].loc[ts, "INTC"]:>10}')
    print(f'   {"ind_above_200":<18} {str(f["ind_above_200"].loc[ts, "INTC"]):>10}')

print()
print('=' * 90)
print('WINNERS vs LOSERS: what separated them?')
print('=' * 90)
tr = pd.read_csv(DATA / 'trades_baseline.csv', parse_dates=['entry_date'])
tr['bucket'] = np.where(tr['excess'] > 0, 'beat SPY', 'lost to SPY')
cols = ['dd_3y', 'rel_12m', 'ret_12m', 'ind_ret_12m', 'vol_ann']
print(tr.groupby('bucket')[cols].median().round(1).to_string())
print()
print('Trade count by bucket:', tr['bucket'].value_counts().to_dict())
