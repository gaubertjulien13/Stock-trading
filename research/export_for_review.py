"""Export the numbers behind the strategy review into a compact JSON payload."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build, setups
from portfolio_test import equity_curve, stats
from tune_momentum import momentum_mask

OUT = DATA / 'review_payload.json'


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    base = build(d, f)
    spy_ret = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0)
    start = pd.Timestamp('2008-01-01')

    payload = {}

    # ---- equity curves: momentum finalist vs the Intel playbook vs SPY
    masks = setups(f, base, d)
    mom = momentum_mask(f, base, close, d=d)

    curves = {}
    port_mom, _ = equity_curve(mom, d, hold_days=126, max_pos=20)
    curves['Momentum (validated)'] = port_mom.loc[start:]
    port_lag, _ = equity_curve(masks['C. Damaged + reclaim (industry agnostic)'], d,
                               hold_days=126, max_pos=20)
    curves['Intel-style turnaround'] = port_lag.loc[start:]
    port_j, _ = equity_curve(masks['J. Laggard turning: damaged + hot industry + rel_3m>0'],
                             d, hold_days=126, max_pos=20)
    curves['Laggard + confirmation'] = port_j.loc[start:]
    curves['SPY buy & hold'] = spy_ret.loc[start:]

    eq = pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()}).dropna()
    eq_m = eq.resample('ME').last()
    payload['equity'] = [
        {'date': d_.strftime('%Y-%m'), **{k: round(float(eq_m.loc[d_, k]), 3) for k in eq_m.columns}}
        for d_ in eq_m.index
    ]

    payload['portfolio_stats'] = []
    for k, v in curves.items():
        s = stats(v, spy_ret.loc[start:], k)
        payload['portfolio_stats'].append({kk: (round(float(vv), 2) if isinstance(vv, (int, float, np.floating)) else vv)
                                           for kk, vv in s.items() if kk != 'avg_positions'})

    # ---- setup edge table
    hr = pd.read_csv(DATA / 'horse_race.csv')
    keep = ['setup', 'n_signals', '6m_med_ret', '6m_med_exSPY', '6m_beatSPY%', '12m_med_exSPY']
    payload['horse_race'] = hr[keep].round(2).replace({np.nan: None}).to_dict('records')

    # ---- factor power
    payload['factors'] = [
        {'factor': 'off_52w_low_pct', 'ic': 0.11, 'spread': 7.27, 'stable': False,
         'note': 'Strongest, but sign flips in 2023-2026'},
        {'factor': 'vol_ann', 'ic': 0.06, 'spread': 5.50, 'stable': False,
         'note': 'Just rewards beta; flips sign recently'},
        {'factor': 'ret_6m', 'ic': 0.05, 'spread': 3.48, 'stable': False, 'note': 'Noise'},
        {'factor': 'rel_12m (the laggard factor)', 'ic': 0.02, 'spread': 2.92, 'stable': False,
         'note': 'Core of the Intel thesis. No predictive power.'},
        {'factor': 'dd_3y (beaten-down-ness)', 'ic': 0.01, 'spread': 0.62, 'stable': False,
         'note': 'Core of the Intel thesis. No predictive power.'},
        {'factor': 'sma200_slope', 'ic': 0.02, 'spread': 1.34, 'stable': False, 'note': 'Noise'},
    ]

    # ---- INTC path
    intc = close['INTC'].dropna()
    intc_m = intc.resample('ME').last().loc['2021':]
    peak3y = intc.rolling(756, min_periods=200).max()
    dd = (intc / peak3y - 1) * 100
    dd_m = dd.resample('ME').last().loc['2021':]
    payload['intc'] = [{'date': i.strftime('%Y-%m'), 'price': round(float(v), 2),
                        'dd': round(float(dd_m.get(i, np.nan)), 1) if pd.notna(dd_m.get(i, np.nan)) else None}
                       for i, v in intc_m.items()]

    # ---- holdout results
    payload['holdout'] = [
        {'strategy': 'Momentum top20%, 20 pos, 6m hold', 'train_excess': 1.43,
         'test_cagr': 17.78, 'test_spy': 14.48, 'test_excess': 3.30, 'test_sharpe': 0.96,
         'test_maxdd': -27.1, 'spy_maxdd': -33.7},
        {'strategy': '+ SPY regime filter', 'train_excess': 2.08, 'test_cagr': 8.44,
         'test_spy': 14.48, 'test_excess': -6.04, 'test_sharpe': 0.53, 'test_maxdd': -35.2,
         'spy_maxdd': -33.7},
        {'strategy': '+ volatility cap', 'train_excess': 2.18, 'test_cagr': 16.35,
         'test_spy': 14.48, 'test_excess': 1.87, 'test_sharpe': 0.90, 'test_maxdd': -27.1,
         'spy_maxdd': -33.7},
        {'strategy': 'Momentum top10% (more selective)', 'train_excess': -0.01,
         'test_cagr': 14.07, 'test_spy': 14.48, 'test_excess': -0.41, 'test_sharpe': 0.73,
         'test_maxdd': -29.4, 'spy_maxdd': -33.7},
        {'strategy': 'Momentum 6m lookback, 3m hold', 'train_excess': -4.20,
         'test_cagr': 11.65, 'test_spy': 14.48, 'test_excess': -2.83, 'test_sharpe': 0.62,
         'test_maxdd': -28.6, 'spy_maxdd': -33.7},
    ]

    # ---- year by year for the finalist
    port = curves['Momentum (validated)']
    sp = spy_ret.reindex(port.index)
    yr = pd.DataFrame({'s': port, 'b': sp}).groupby(port.index.year).apply(
        lambda g: pd.Series({'strategy': ((1 + g['s']).prod() - 1) * 100,
                             'spy': ((1 + g['b']).prod() - 1) * 100}))
    payload['yearly'] = [{'year': int(i), 'strategy': round(float(r['strategy']), 1),
                          'spy': round(float(r['spy']), 1),
                          'excess': round(float(r['strategy'] - r['spy']), 1)}
                         for i, r in yr.iterrows()]

    payload['coverage'] = {
        'universe_total': 841, 'current_members': 503, 'left_index': 338,
        'recovered': 158, 'blind': 180,
        'blind_examples': ['LEH', 'SIVB', 'FRC', 'FNM', 'FRE', 'JCP', 'EK', 'ABK'],
    }

    OUT.write_text(json.dumps(payload, indent=1))
    print(f'Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)')
    for k, v in payload.items():
        print(f'  {k}: {len(v) if isinstance(v, (list, dict)) else v}')


if __name__ == '__main__':
    main()
