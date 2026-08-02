"""Compact JSON for the strategy review canvas. Regenerates every number shown."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA
from horse_race import build, setups
from portfolio_test import stats
from tune_momentum import momentum_mask
from verify import delisting_returns, equity_curve

START = pd.Timestamp('2008-01-01')
OUT = DATA / 'canvas_data.json'


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close = d['close']
    base = build(d, f)
    rets, _ = delisting_returns(close, 0.0)

    spy_ret = d['etf']['SPY'].pct_change(fill_method=None).fillna(0.0)
    rsp = d['etf']['RSP'].reindex(close.index).ffill()
    rsp_ret = rsp.pct_change(fill_method=None).fillna(0.0)

    masks = setups(f, base, d)
    mom = momentum_mask(f, base, close, d=d)

    curves = {}
    p, _ = equity_curve(mom, d, rets, max_pos=50, seed=0)
    curves['Momentum (50 names)'] = p.loc[START:]
    p, _ = equity_curve(masks['J. Laggard turning: damaged + hot industry + rel_3m>0'],
                        d, rets, max_pos=20)
    curves['Intel playbook'] = p.loc[START:]
    p, _ = equity_curve(masks['C. Damaged + reclaim (industry agnostic)'], d, rets, max_pos=20)
    curves['Beaten down + reclaim'] = p.loc[START:]
    curves['S&P 500 (SPY)'] = spy_ret.loc[START:]
    curves['Equal-weight (RSP)'] = rsp_ret.loc[START:]

    payload = {}

    eq = pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()}).dropna()
    ye = eq.resample('YE').last()
    ye = pd.concat([pd.DataFrame({c: [1.0] for c in eq.columns},
                                 index=[pd.Timestamp('2007-12-31')]), ye])
    payload['equity'] = [{'year': str(i.year),
                          **{k: round(float(ye.loc[i, k]), 2) for k in ye.columns}}
                         for i in ye.index]

    payload['portfolio'] = []
    for k, v in curves.items():
        if k in ('S&P 500 (SPY)', 'Equal-weight (RSP)'):
            continue
        s = stats(v, spy_ret.loc[START:], k)
        sr = stats(v, rsp_ret.loc[START:], k)
        payload['portfolio'].append({
            'strategy': k, 'cagr': round(s['CAGR_%'], 1),
            'vs_spy': round(s['excess_%'], 1),
            'vs_rsp': round(s['CAGR_%'] - sr['SPY_CAGR_%'], 1),
            'sharpe': round(s['sharpe'], 2), 'maxdd': round(s['maxDD_%'], 0),
            'x': round(s['final_x'], 1)})
    s = stats(spy_ret.loc[START:], spy_ret.loc[START:], 'S&P 500 (SPY)')
    sr = stats(rsp_ret.loc[START:], spy_ret.loc[START:], 'Equal-weight (RSP)')
    for lbl, st in (('S&P 500 (SPY)', s), ('Equal-weight (RSP)', sr)):
        payload['portfolio'].append({
            'strategy': lbl, 'cagr': round(st['CAGR_%'], 1),
            'vs_spy': round(st['excess_%'], 1), 'vs_rsp': None,
            'sharpe': round(st['sharpe'], 2), 'maxdd': round(st['maxDD_%'], 0),
            'x': round(st['final_x'], 1)})

    for name, path in (('concentration', 'verify_concentration.csv'),
                       ('exits', 'verify_exits.csv'),
                       ('selection', 'verify_selection.csv')):
        payload[name] = pd.read_csv(DATA / path).round(2).replace({np.nan: None}).to_dict('records')

    ten = pd.read_csv(DATA / 'verify_10name.csv')
    payload['tenname'] = {
        'mean': round(float(ten['vs_SPY'].mean()), 2),
        'std': round(float(ten['vs_SPY'].std()), 2),
        'min': round(float(ten['vs_SPY'].min()), 2),
        'max': round(float(ten['vs_SPY'].max()), 2),
        'beat': int((ten['vs_SPY'] > 0).sum()), 'n': len(ten),
        'draws': [round(float(x), 2) for x in ten['vs_SPY']]}

    yr = pd.read_csv(DATA / 'verify_yearly50.csv', index_col=0)
    payload['yearly'] = [{'year': str(int(i)), 'strategy': round(float(r['strategy_%']), 1),
                          'spy': round(float(r['spy_%']), 1),
                          'vs_spy': round(float(r['vs_SPY']), 1)} for i, r in yr.iterrows()]

    intc = close['INTC'].dropna().resample('ME').last().loc['2021':]
    payload['intc'] = [{'date': i.strftime('%Y-%m'), 'price': round(float(v), 2)}
                       for i, v in intc.items()]

    OUT.write_text(json.dumps(payload, indent=1))
    print(f'Wrote {OUT}')
    for k, v in payload.items():
        print(f'  {k}: {len(v)}')


if __name__ == '__main__':
    main()
