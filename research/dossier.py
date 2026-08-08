"""Research dossiers for Stage-1 candidates: the evidence layer behind a decision.

Why this exists. `validate_stage2.py` showed the mechanical pillars (1, 2, 3, 6)
carry no rank information, and `ANALOGS.md` reached the same place from the other
side - winners and losers are indistinguishable at onset on price features:

    "Price and insider data get you into the right room. They do not pick the chair."

Pillars 4 and 5 - a concrete recovery mechanism and outside recognition - are the
ones the analog study credits, and the only ones never tested, because they need
information that does not exist in a price panel. This script gathers that
information: point-in-time fundamentals from SEC XBRL, insider clusters, recent
headlines, and how the name sits against its actual industry peers.

It deliberately does not score or rank on any of it. The output is evidence for a
human decision, and `journal.py` is what eventually measures whether that human
decision adds value.

    venv/bin/python3 research/dossier.py --top 12
    venv/bin/python3 research/dossier.py --top 12 --no-news    # skip yfinance calls
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_fundamentals
from engine import DATA
from recommend import fetch_headlines, stage1_mask, stage2

PEER_MIN = 4


def _fmt_money(v, unit='B'):
    if not np.isfinite(v):
        return 'n/a'
    div = 1e9 if unit == 'B' else 1e6
    return f"${v / div:,.2f}{unit}"


def _fmt_pct(v, dp=1, sign=True):
    if not np.isfinite(v):
        return 'n/a'
    return f"{v:+.{dp}f}%" if sign else f"{v:.{dp}f}%"


def _fmt_pp(v, dp=1):
    """Percentage-point deltas: '+2.8pp', never '+2.8%pp'."""
    return f"{v:+.{dp}f}pp" if np.isfinite(v) else 'n/a'


def load_panel():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        return pickle.load(fh)


def load_insiders():
    p = DATA / 'insider_buys.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    return df


def insider_summary(buys, ticker, asof, windows=(90, 180)):
    """Form 4 open-market buying. ANALOGS found only 26% of winners had any, so
    absence is uninformative - presence is a mild positive, not a requirement."""
    if buys is None:
        return {}
    out = {}
    for w in windows:
        sub = buys[(buys['ticker'] == ticker) &
                   (buys['filing_date'] >= asof - pd.Timedelta(days=w)) &
                   (buys['filing_date'] <= asof)]
        out[f'{w}d'] = {
            'filings': int(sub['n_filings'].sum()) if not sub.empty else 0,
            'insiders': int(sub['n_insiders'].sum()) if not sub.empty else 0,
            'value': float(sub['value'].sum()) if not sub.empty else 0.0,
        }
    return out


def peer_context(f, d, ticker, asof):
    """Is the damage company-specific or industry-wide?

    The Intel thesis needs a healthy industry and a broken company. If peers are
    down just as much, this is a sector problem and the setup does not apply.
    """
    try:
        ind = str(f['industry'].at[asof, ticker])
    except Exception:
        return {}
    row_ind = f['industry'].loc[asof]
    peers = row_ind[row_ind == ind].index
    member = d['member'].reindex(columns=d['close'].columns).fillna(False).loc[asof]
    peers = [p for p in peers if p != ticker and bool(member.get(p, False))]
    if len(peers) < PEER_MIN:
        return {'industry': ind, 'n_peers': len(peers)}
    r12 = f['ret_12m'].loc[asof]
    peer_r = r12[peers].dropna()
    me = float(r12.get(ticker, np.nan))
    if peer_r.empty or not np.isfinite(me):
        return {'industry': ind, 'n_peers': len(peers)}
    return {
        'industry': ind,
        'n_peers': int(len(peer_r)),
        'peer_median_12m': float(peer_r.median()),
        'ticker_12m': me,
        'vs_peer_median_pp': me - float(peer_r.median()),
        'percentile_in_peers': float((peer_r < me).mean() * 100),
        'peers_also_down': int((peer_r < -20).sum()),
    }


def pillar_checklist(row, fund, peers, ins):
    """The ANALOGS six-pillar frame, with evidence attached and gaps named.

    Pillars 1-3 and 6 are answerable from data; validate_stage2 showed they do not
    rank returns, so they appear as context, not as a verdict. Pillars 4-5 are left
    open on purpose - they are the human's job and the reason this file exists.
    """
    out = []

    ind_ok = np.isfinite(row.get('ind_ret_12m', np.nan)) and row['ind_ret_12m'] > 0
    out.append({
        'pillar': '1. Structurally growing industry',
        'evidence': (f"{peers.get('industry', '?')} 12m {_fmt_pct(row.get('ind_ret_12m', np.nan))}"
                     + (f", peers median 12m {_fmt_pct(peers['peer_median_12m'])}"
                        if 'peer_median_12m' in peers else '')),
        'auto': 'yes' if ind_ok else 'no',
    })

    rev_g = fund.get('revenue_3y_pct', np.nan) if fund else np.nan
    out.append({
        'pillar': '2. Franchise / strategic asset',
        'evidence': (f"revenue TTM {_fmt_money(fund.get('revenue_ttm', np.nan))}, "
                     f"3y {_fmt_pct(rev_g)}, gross margin "
                     f"{_fmt_pct(fund.get('gross_margin_pct', np.nan), sign=False)}"
                     if fund else 'no fundamentals'),
        'auto': 'judge',
    })

    if fund:
        rev, om = fund.get('revenue_yoy_pct', np.nan), fund.get('operating_margin_delta_pp', np.nan)
        healing = (np.isfinite(rev) and rev > 0) and (np.isfinite(om) and om > 0)
        secular = (np.isfinite(rev) and rev < -5) and (np.isfinite(om) and om < 0)
        verdict = 'healing' if healing else ('deteriorating' if secular else 'mixed')
        ev = (f"revenue YoY {_fmt_pct(rev)}, operating margin "
              f"{_fmt_pct(fund.get('operating_margin_pct', np.nan), sign=False)} "
              f"({_fmt_pp(om)} YoY)")
    else:
        verdict, ev = 'unknown', 'no fundamentals'
    out.append({
        'pillar': '3. Damage temporary, not secular',
        'evidence': ev + f"; price {_fmt_pct(row.get('dd_3y_pct', np.nan))} off 3y peak, "
                         f"{_fmt_pct(row.get('off_52w_low_pct', np.nan), sign=False)} off 52w low",
        'auto': verdict,
    })

    out.append({
        'pillar': '4. Concrete recovery mechanism',
        'evidence': 'see headlines below - name the specific mechanism, or pass',
        'auto': 'JUDGE',
    })
    out.append({
        'pillar': '5. Outside recognition',
        'evidence': 'see headlines below - is anyone else articulating this thesis?',
        'auto': 'JUDGE',
    })

    if fund:
        fcf = fund.get('fcf_ttm', np.nan)
        nd = fund.get('net_debt_to_fcf', np.nan)
        risky = (np.isfinite(fcf) and fcf < 0) or (np.isfinite(nd) and nd > 6)
        netdebt = fund.get('net_debt', np.nan)
        ev6 = (f"FCF {_fmt_money(fcf)}, cash {_fmt_money(fund.get('cash', np.nan))}, "
               + (f"net cash {_fmt_money(abs(netdebt))}"
                  if np.isfinite(netdebt) and netdebt < 0
                  else f"net debt {_fmt_money(netdebt)}")
               + (f" ({nd:.1f}x FCF)" if np.isfinite(nd) and nd > 0 else '')
               + (f", diluting {fund['dilution_yoy_pct']:.1f}%/yr"
                  if np.isfinite(fund.get('dilution_yoy_pct', np.nan))
                  and fund['dilution_yoy_pct'] > 3 else ''))
        auto6 = 'at risk' if risky else 'adequate'
    else:
        ev6, auto6 = 'no fundamentals', 'unknown'
    ins90 = ins.get('90d', {}).get('insiders', 0) if ins else 0
    if ins90:
        ev6 += f"; {ins90} insider buyer(s) in 90d"
    out.append({'pillar': '6. Survivability', 'evidence': ev6, 'auto': auto6})
    return out


def build(asof=None, top=12, with_news=True, news_items=8):
    d, f = load_panel()
    close = d['close']
    asof = (pd.Timestamp(asof) if asof else close.index[-1])
    if asof not in close.index:
        asof = close.index[close.index <= asof][-1]

    cands = stage1_mask(f, d, asof)
    print(f"  Stage 1 candidates on {asof.date()}: {len(cands)}")
    scored = stage2(cands, asof, f, d, buys=None, fetch_news=False)
    if scored.empty:
        return None, None

    # Rank by divergence from industry, not by the Stage-2 score: the score was
    # shown to carry no forward information, while divergence is what defines the setup.
    scored = scored.sort_values('rel_12m').head(top).reset_index(drop=True)
    tickers = scored['ticker'].tolist()
    print(f"  building dossiers for {len(tickers)}: {', '.join(tickers)}")

    print("  fetching SEC fundamentals...")
    funds = fetch_fundamentals.get(tickers, quiet=True)
    buys = load_insiders()

    entries = []
    for i, row in scored.iterrows():
        t = row['ticker']
        fund = funds.get(t)
        peers = peer_context(f, d, t, asof)
        ins = insider_summary(buys, t, asof)
        heads = []
        if with_news:
            print(f"\r  headlines {i + 1}/{len(scored)}  {t:6s}", end='', flush=True)
            heads = fetch_headlines(t, max_items=news_items)
        entries.append({
            'ticker': t,
            'price': float(row['price']),
            'score': float(row['score']),
            'band': row['band'],
            'tag': row['tag'],
            'dd_3y_pct': float(row['dd_3y_pct']),
            'rel_12m': float(row['rel_12m']),
            'rel_3m': float(row['rel_3m']),
            'off_52w_low_pct': float(row['off_52w_low_pct']),
            'ind_ret_12m': float(row['ind_ret_12m']),
            'above_50': bool(row['above_50']),
            'above_200': bool(row['above_200']),
            'vol_ann': float(row['vol_ann']),
            'fundamentals': fund,
            'peers': peers,
            'insiders': ins,
            'headlines': heads,
            'pillars': pillar_checklist(row, fund, peers, ins),
        })
    if with_news:
        print("\r" + " " * 40 + "\r", end='')
    return asof, entries


def render(asof, entries):
    L = [f'# Research dossiers — {asof.date()}', '',
         f'{len(entries)} candidates, ranked by divergence from their industry '
         f'(most negative `rel_12m` first).', '',
         '> The Stage-2 score is shown for continuity only. `validate_stage2.py` found it '
         'carries no\n> forward rank information (IC ≈ 0, quintiles sloping the wrong way), '
         'so do not treat\n> STRONG/WATCH as a quality ranking. Pillars 4 and 5 are the '
         'decision; everything else is context.', '']

    L += ['| # | Ticker | Price | DD 3y | vs industry | Revenue YoY | Op margin Δ | FCF | Survivability |',
          '|---|---|---|---|---|---|---|---|---|']
    for i, e in enumerate(entries, 1):
        fu = e['fundamentals'] or {}
        surv = next((p['auto'] for p in e['pillars'] if p['pillar'].startswith('6')), '?')
        L.append(f"| {i} | {e['ticker']} | ${e['price']:.2f} | {e['dd_3y_pct']:.0f}% | "
                 f"{e['rel_12m']:.0f}pp | {_fmt_pct(fu.get('revenue_yoy_pct', np.nan))} | "
                 f"{_fmt_pp(fu.get('operating_margin_delta_pp', np.nan))} | "
                 f"{_fmt_money(fu.get('fcf_ttm', np.nan))} | {surv} |")
    L.append('')

    for i, e in enumerate(entries, 1):
        fu, pe, ins = e['fundamentals'], e['peers'], e['insiders']
        L += ['---', '', f"## {i}. {e['ticker']} — ${e['price']:.2f}", '',
              f"**Setup.** {e['dd_3y_pct']:.0f}% off its 3-year peak, "
              f"{e['rel_12m']:.0f}pp behind its industry over 12 months, "
              f"{e['off_52w_low_pct']:.0f}% off the 52-week low. "
              f"{'Above' if e['above_200'] else 'Below'} the 200dma, "
              f"{'above' if e['above_50'] else 'below'} the 50dma. "
              f"Annualized vol {e['vol_ann']:.0f}%. Tag `{e['tag']}`.", '']

        if pe.get('n_peers', 0) >= PEER_MIN:
            L += [f"**Industry.** {pe['industry']} up {e['ind_ret_12m']:.0f}% over 12m. "
                  f"Against {pe['n_peers']} index peers, {e['ticker']} returned "
                  f"{pe['ticker_12m']:.0f}% vs a peer median of {pe['peer_median_12m']:.0f}% "
                  f"({pe['vs_peer_median_pp']:+.0f}pp, {pe['percentile_in_peers']:.0f}th percentile). "
                  + (f"{pe['peers_also_down']} of {pe['n_peers']} peers are also down more than 20%, "
                     "so check whether this is really company-specific."
                     if pe['peers_also_down'] >= max(2, pe['n_peers'] // 3)
                     else "Peers are holding up, so the damage looks company-specific."), '']
        else:
            L += [f"**Industry.** {pe.get('industry', '?')} up {e['ind_ret_12m']:.0f}% over 12m "
                  f"(too few index peers for a comparison).", '']

        if fu:
            L += ['**Financials** (SEC XBRL, period end '
                  f"{fu.get('period_end')}):", '',
                  f"- Revenue TTM {_fmt_money(fu['revenue_ttm'])}, "
                  f"YoY {_fmt_pct(fu['revenue_yoy_pct'])}, 3y {_fmt_pct(fu['revenue_3y_pct'])}",
                  f"- Gross margin {_fmt_pct(fu['gross_margin_pct'], sign=False)} "
                  f"({_fmt_pp(fu['gross_margin_delta_pp'])} YoY), operating margin "
                  f"{_fmt_pct(fu['operating_margin_pct'], sign=False)} "
                  f"({_fmt_pp(fu['operating_margin_delta_pp'])} YoY)",
                  f"- FCF TTM {_fmt_money(fu['fcf_ttm'])} "
                  f"(margin {_fmt_pct(fu['fcf_margin_pct'], sign=False)}), "
                  f"cash {_fmt_money(fu['cash'])}, net debt {_fmt_money(fu['net_debt'])}",
                  f"- Flags: {', '.join(fu['flags']) if fu['flags'] else 'none'}", '']
        else:
            L += ['**Financials.** Not available from SEC XBRL for this ticker.', '']

        if ins:
            a, b = ins.get('90d', {}), ins.get('180d', {})
            if a.get('insiders') or b.get('insiders'):
                L += [f"**Insiders.** {a.get('insiders', 0)} buyer(s) / "
                      f"{_fmt_money(a.get('value', 0), 'M')} in 90d; "
                      f"{b.get('insiders', 0)} / {_fmt_money(b.get('value', 0), 'M')} in 180d.", '']
            else:
                L += ['**Insiders.** No open-market buying in 180 days. '
                      '(Only 26% of historical winners had any, so this is weak evidence.)', '']

        L.append('**Recent headlines.**')
        if e['headlines']:
            L += [''] + [f'- {h}' for h in e['headlines']] + ['']
        else:
            L += ['', '- none retrieved', '']

        L += ['**Pillar checklist.**', '',
              '| Pillar | Read | Evidence |', '|---|---|---|']
        for p in e['pillars']:
            L.append(f"| {p['pillar']} | `{p['auto']}` | {p['evidence']} |")
        L += ['', '**Your call.** Mechanism: ______  ·  Who else sees it: ______  ·  '
                  'Decision: buy / watch / pass  ·  Why: ______', '']

    L += ['---', '', '## How to use this',
          '', '1. Read pillars 4 and 5 for each name. If you cannot name a specific recovery '
          'mechanism in one sentence, pass — that is the pillar the analog study credits.',
          '2. Check pillar 3 against the price story. Revenue falling with margins compressing '
          'is secular decline wearing a turnaround costume.',
          '3. Log the decision with `journal.py` including your one-sentence thesis, so the '
          'judgment layer can actually be measured over 20–30 decisions.',
          '4. Size small and spread across names. The historical payoff is venture-shaped: '
          'the median candidate loses money and a few return +150% or more.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--top', type=int, default=12, help='How many candidates (default 12)')
    ap.add_argument('--asof', default=None, help='Freeze to a past date (YYYY-MM-DD)')
    ap.add_argument('--no-news', action='store_true', help='Skip yfinance headline calls')
    ap.add_argument('--out', default=None, help='Output markdown path')
    args = ap.parse_args()

    asof, entries = build(asof=args.asof, top=args.top, with_news=not args.no_news)
    if not entries:
        print("no candidates")
        return

    stamp = asof.strftime('%Y%m%d')
    md_path = Path(args.out) if args.out else DATA / f'dossier_{stamp}.md'
    json_path = DATA / f'dossier_{stamp}.json'
    md_path.write_text(render(asof, entries))
    json_path.write_text(json.dumps(
        {'asof': str(asof.date()), 'generated': datetime.now().isoformat(),
         'entries': entries}, indent=2, default=str))
    print(f"\n  wrote {md_path}")
    print(f"  wrote {json_path}")
    print("\n  Next: read the dossier, then log decisions with:")
    print(f"    venv/bin/python3 research/journal.py add --ticker XYZ --decision buy "
          f"--thesis \"...\"")


if __name__ == '__main__':
    main()
