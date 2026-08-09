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

The first run builds a company-profile cache over the whole index (~2 minutes);
later runs reuse it and refresh entries older than a month.
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
PROFILES = DATA / 'company_profiles.json'
PROFILE_MAX_AGE_DAYS = 30
# Bump when fields are added, so existing caches backfill instead of silently
# rendering blanks for everything cached before the change.
PROFILE_VERSION = 2


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


def _ord(n):
    """'1st', '33rd' - percentiles read as text, not as '1th'."""
    n = int(round(n))
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


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


def _load_profiles():
    if not PROFILES.exists():
        return {}
    try:
        return json.loads(PROFILES.read_text())
    except Exception:
        return {}


def fetch_profiles(tickers, refresh=False, quiet=False):
    """Company descriptors from yfinance, cached to disk.

    Needed for the things a price panel cannot say: what the company sells, which
    industry it really competes in (the panel's label is a return-correlation
    bucket, not a business), and where analysts think the price is going.

    Cached because the competitor lookup needs the whole index, which is ~2
    minutes of calls on a cold cache and instant afterwards. Entries older than
    PROFILE_MAX_AGE_DAYS are refetched so market caps and targets do not rot.
    """
    import yfinance as yf

    cache = _load_profiles()
    cutoff = (datetime.now() - pd.Timedelta(days=PROFILE_MAX_AGE_DAYS)).isoformat()
    todo = [t for t in tickers
            if refresh or t not in cache
            or cache[t].get('v') != PROFILE_VERSION
            or (cache[t].get('fetched') or '') < cutoff]

    for i, t in enumerate(todo, 1):
        if not quiet and (i % 25 == 0 or i == len(todo)):
            print(f"\r  profiles {i}/{len(todo)}", end='', flush=True)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        cache[t] = {
            'long_name': info.get('longName') or info.get('shortName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'summary': info.get('longBusinessSummary'),
            'market_cap': info.get('marketCap'),
            'ps_ttm': info.get('priceToSalesTrailing12Months'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'trailing_eps': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_q_growth': info.get('earningsQuarterlyGrowth'),
            'target_mean': info.get('targetMeanPrice'),
            'target_low': info.get('targetLowPrice'),
            'target_high': info.get('targetHighPrice'),
            'target_n': info.get('numberOfAnalystOpinions'),
            'recommendation': info.get('recommendationKey'),
            'fetched': datetime.now().isoformat(),
            'v': PROFILE_VERSION,
        }
    if todo:
        PROFILES.write_text(json.dumps(cache, indent=1, default=str))
        if not quiet:
            print(f"\r  profiles: {len(todo)} fetched, {len(cache)} cached      ")
    return cache


def _first_sentence(text, limit=260):
    """yfinance summaries open with a usable one-line definition of the business."""
    if not text:
        return None
    s = ' '.join(str(text).split())
    cut = s.find('. ')
    out = s[:cut + 1] if cut > 40 else s
    return out if len(out) <= limit else out[:limit].rsplit(' ', 1)[0] + '…'


def industry_context(ticker, profiles, members, ret12):
    """Real industry peers, as opposed to the panel's correlation bucket.

    The panel assigns industries by return correlation against sector ETFs, which
    puts ORCL and CRM in XLF. That is fine for measuring divergence but useless
    for naming competitors, so this uses the reported industry classification.
    """
    me = profiles.get(ticker) or {}
    ind = me.get('industry')
    if not ind:
        return {}
    pool = []
    for t in members:
        if t == ticker:
            continue
        p = profiles.get(t) or {}
        if p.get('industry') != ind or not p.get('market_cap'):
            continue
        pool.append((float(p['market_cap']), t, p))
    pool.sort(reverse=True)

    ps = [float(p['ps_ttm']) for _, _, p in pool
          if isinstance(p.get('ps_ttm'), (int, float)) and p['ps_ttm'] and p['ps_ttm'] > 0]
    comps = [{'ticker': t, 'name': p.get('long_name'), 'market_cap': mc,
              'ret_12m': float(ret12.get(t, np.nan)), 'ps_ttm': p.get('ps_ttm')}
             for mc, t, p in pool[:3]]
    return {
        'sector': me.get('sector'),
        'industry': ind,
        'n_in_industry': len(pool),
        'competitors': comps,
        'peer_ps_median': float(np.median(ps)) if len(ps) >= 3 else np.nan,
        'pool': [t for _, t, _ in pool],
    }


def _nums(profiles, tickers, key, lo=-np.inf, hi=np.inf):
    out = []
    for t in tickers:
        v = (profiles.get(t) or {}).get(key)
        if isinstance(v, (int, float)) and v is not None and np.isfinite(v) and lo < v < hi:
            out.append(float(v))
    return out


def eps_context(ticker, profiles, pool):
    """Earnings against the industry the company actually competes in.

    Revenue and margins already appear from SEC filings; this adds the bottom
    line and what the market pays for it. EPS levels are not comparable across
    companies with different share counts, so the comparison is on growth and on
    the multiple - both of which say whether the market is discounting this name
    specifically or the whole industry.

    yfinance for both sides on purpose: a SEC-derived figure for the candidate
    against a vendor figure for peers would not be measuring the same thing.
    """
    me = profiles.get(ticker) or {}
    if not pool:
        return {}

    # P/E outliers are usually near-zero earnings, not information.
    peer_g = _nums(profiles, pool, 'earnings_growth', -5, 10)
    peer_pe = _nums(profiles, pool, 'trailing_pe', 0, 200)
    peer_fpe = _nums(profiles, pool, 'forward_pe', 0, 200)

    g = me.get('earnings_growth')
    g = float(g) if isinstance(g, (int, float)) and g is not None and np.isfinite(g) else np.nan
    out = {
        'trailing_eps': me.get('trailing_eps'),
        'forward_eps': me.get('forward_eps'),
        'eps_growth_pct': g * 100 if np.isfinite(g) else np.nan,
        'trailing_pe': me.get('trailing_pe'),
        'forward_pe': me.get('forward_pe'),
        'n_peers': len(peer_g),
        'peer_eps_growth_pct': float(np.median(peer_g)) * 100 if len(peer_g) >= 3 else np.nan,
        'peer_trailing_pe': float(np.median(peer_pe)) if len(peer_pe) >= 3 else np.nan,
        'peer_forward_pe': float(np.median(peer_fpe)) if len(peer_fpe) >= 3 else np.nan,
    }
    if np.isfinite(g) and len(peer_g) >= 3:
        out['eps_growth_vs_peers_pp'] = out['eps_growth_pct'] - out['peer_eps_growth_pct']
        out['eps_growth_percentile'] = float((np.array(peer_g) < g).mean() * 100)

    te, pe_med = me.get('trailing_pe'), out['peer_trailing_pe']
    if (isinstance(te, (int, float)) and te and te > 0 and np.isfinite(pe_med) and pe_med > 0):
        out['pe_discount_pct'] = (float(te) / pe_med - 1) * 100
    fe, fpe_med = me.get('forward_pe'), out['peer_forward_pe']
    if (isinstance(fe, (int, float)) and fe and fe > 0 and np.isfinite(fpe_med) and fpe_med > 0):
        out['fpe_discount_pct'] = (float(fe) / fpe_med - 1) * 100
    return out


def _sparkline(values):
    blocks = '▁▂▃▄▅▆▇█'
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return ''
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return blocks[0] * v.size
    idx = np.round((v - lo) / (hi - lo) * (len(blocks) - 1)).astype(int)
    return ''.join(blocks[i] for i in idx)


def price_trend(close, ticker, asof, months=12):
    """Shape of the last 12 months, from the panel so it respects --asof.

    A single drawdown number cannot distinguish a name that has been bleeding all
    year from one that bottomed six months ago and is recovering, and that
    difference is most of what pillar 3 is asking about.
    """
    s = close[ticker].loc[:asof].dropna()
    win = s.loc[s.index >= asof - pd.DateOffset(months=months)]
    if len(win) < 30:
        return {}
    weekly = win.resample('W').last().dropna()
    hi_dt, lo_dt = win.idxmax(), win.idxmin()
    last = float(win.iloc[-1])

    legs = []
    for k in range(4, 0, -1):
        a = asof - pd.DateOffset(months=3 * k)
        b = asof - pd.DateOffset(months=3 * (k - 1))
        seg = s.loc[(s.index > a) & (s.index <= b)]
        if len(seg) > 5:
            legs.append({'label': f"{a:%b}–{b:%b}",
                         'ret_pct': (float(seg.iloc[-1]) / float(seg.iloc[0]) - 1) * 100})
    return {
        'sparkline': _sparkline(weekly.values),
        'ret_12m_pct': (last / float(win.iloc[0]) - 1) * 100,
        'high': float(win.max()), 'high_date': str(hi_dt.date()),
        'low': float(win.min()), 'low_date': str(lo_dt.date()),
        'from_high_pct': (last / float(win.max()) - 1) * 100,
        'from_low_pct': (last / float(win.min()) - 1) * 100,
        'legs': legs,
        'bottom_is_recent': bool((asof - lo_dt).days <= 90),
    }


def reference_levels(ticker, price, close, asof, profiles, ind):
    """Sell-side anchors. Not forecasts, and deliberately more than one.

    Three independent references disagreeing is the useful signal; a single
    number would imply a precision none of these methods has. See the caveats
    printed alongside them in the rendered dossier.
    """
    prof = profiles.get(ticker) or {}
    out = {}

    s = close[ticker].loc[:asof].dropna()
    w3 = s.loc[s.index >= asof - pd.DateOffset(years=3)]
    if len(w3) > 60:
        peak = float(w3.max())
        out['peak_3y'] = peak
        out['halfway_to_peak'] = price + 0.5 * (peak - price)
        out['upside_to_peak_pct'] = (peak / price - 1) * 100
        out['upside_halfway_pct'] = (out['halfway_to_peak'] / price - 1) * 100

    tm = prof.get('target_mean')
    if isinstance(tm, (int, float)) and tm and tm > 0:
        out['analyst_mean'] = float(tm)
        out['analyst_low'] = prof.get('target_low')
        out['analyst_high'] = prof.get('target_high')
        out['analyst_n'] = prof.get('target_n')
        out['upside_analyst_pct'] = (float(tm) / price - 1) * 100

    own_ps, med_ps = prof.get('ps_ttm'), (ind or {}).get('peer_ps_median', np.nan)
    if (isinstance(own_ps, (int, float)) and own_ps and own_ps > 0
            and np.isfinite(med_ps) and med_ps > 0):
        out['own_ps'] = float(own_ps)
        out['peer_ps_median'] = float(med_ps)
        out['peer_implied'] = price * float(med_ps) / float(own_ps)
        out['upside_peer_pct'] = (float(med_ps) / float(own_ps) - 1) * 100
    return out


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


def build(asof=None, top=12, with_news=True, news_items=8, refresh_profiles=False):
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

    # Profiles cover the whole index, not just the candidates: naming a company's
    # three largest competitors means knowing every index member's industry.
    member_row = d['member'].reindex(columns=close.columns).fillna(False).loc[asof]
    members = sorted(set(member_row[member_row].index.tolist()) | set(tickers))
    profiles = fetch_profiles(members, refresh=refresh_profiles)
    ret12 = f['ret_12m'].loc[asof]

    entries = []
    for i, row in scored.iterrows():
        t = row['ticker']
        fund = funds.get(t)
        peers = peer_context(f, d, t, asof)
        prof = profiles.get(t) or {}
        ind = industry_context(t, profiles, members, ret12)
        eps = eps_context(t, profiles, ind.get('pool') or [])
        trend = price_trend(close, t, asof)
        levels = reference_levels(t, float(row['price']), close, asof, profiles, ind)
        ins = insider_summary(buys, t, asof)
        heads = []
        if with_news:
            print(f"\r  headlines {i + 1}/{len(scored)}  {t:6s}", end='', flush=True)
            heads = fetch_headlines(t, max_items=news_items)
        entries.append({
            'ticker': t,
            'name': prof.get('long_name'),
            'description': _first_sentence(prof.get('summary')),
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
            'industry_context': ind,
            'eps': eps,
            'trend': trend,
            'levels': levels,
            'insiders': ins,
            'headlines': heads,
            'pillars': pillar_checklist(row, fund, peers, ins),
        })
    if with_news:
        print("\r" + " " * 40 + "\r", end='')
    return asof, entries, close.index[-1]


def render(asof, entries, latest=None):
    L = [f'# Research dossiers — {asof.date()}', '',
         f'{len(entries)} candidates, ranked by divergence from their industry '
         f'(most negative `rel_12m` first).', '',
         '> The Stage-2 score is shown for continuity only. `validate_stage2.py` found it '
         'carries no\n> forward rank information (IC ≈ 0, quintiles sloping the wrong way), '
         'so do not treat\n> STRONG/WATCH as a quality ranking. Pillars 4 and 5 are the '
         'decision; everything else is context.', '']

    if latest is not None and asof < latest:
        L += [f'> **Backdated to {asof.date()}, but not entirely.** Prices, trends and peer '
              'comparisons come\n> from the panel and respect the as-of date. Company '
              'descriptions, market caps, analyst\n> targets and P/S ratios are fetched live '
              'and describe today, not that date — so the\n> "Reference levels" section leaks '
              'the future in a historical run and must not be used\n> to evaluate how this '
              'would have looked at the time.', '']

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
        ic, tr, lv = e.get('industry_context') or {}, e.get('trend') or {}, e.get('levels') or {}
        ep = e.get('eps') or {}
        title = f"## {i}. {e['ticker']}"
        if e.get('name'):
            title += f" — {e['name']}"
        L += ['---', '', f"{title} — ${e['price']:.2f}", '']

        if e.get('description'):
            L += [f"**What it does.** {e['description']}", '']

        if ic.get('industry'):
            line = (f"**Industry.** {ic['industry']}"
                    + (f" ({ic['sector']})" if ic.get('sector') else '')
                    + f" — {ic['n_in_industry'] + 1} S&P 500 names compete here.")
            if ic.get('competitors'):
                parts = []
                for c in ic['competitors']:
                    r = c.get('ret_12m', np.nan)
                    parts.append(f"{c['ticker']} ({_fmt_money(c['market_cap'])}"
                                 + (f", 12m {_fmt_pct(r, dp=0)}" if np.isfinite(r) else '')
                                 + ")")
                line += f" Largest competitors: {', '.join(parts)}."
            L += [line, '']

        L += [f"**Setup.** {e['dd_3y_pct']:.0f}% off its 3-year peak, "
              f"{e['rel_12m']:.0f}pp behind its industry over 12 months, "
              f"{e['off_52w_low_pct']:.0f}% off the 52-week low. "
              f"{'Above' if e['above_200'] else 'Below'} the 200dma, "
              f"{'above' if e['above_50'] else 'below'} the 50dma. "
              f"Annualized vol {e['vol_ann']:.0f}%. Tag `{e['tag']}`.", '']

        if pe.get('n_peers', 0) >= PEER_MIN:
            L += [f"**Divergence.** Return-correlation bucket {pe['industry']} is up "
                  f"{e['ind_ret_12m']:.0f}% over 12m. "
                  f"Against {pe['n_peers']} index peers, {e['ticker']} returned "
                  f"{pe['ticker_12m']:.0f}% vs a peer median of {pe['peer_median_12m']:.0f}% "
                  f"({pe['vs_peer_median_pp']:+.0f}pp, {_ord(pe['percentile_in_peers'])} percentile). "
                  + (f"{pe['peers_also_down']} of {pe['n_peers']} peers are also down more than 20%, "
                     "so check whether this is really company-specific."
                     if pe['peers_also_down'] >= max(2, pe['n_peers'] // 3)
                     else "Peers are holding up, so the damage looks company-specific."), '']
        else:
            L += [f"**Divergence.** Return-correlation bucket {pe.get('industry', '?')} is up "
                  f"{e['ind_ret_12m']:.0f}% over 12m "
                  f"(too few index peers for a comparison).", '']

        if tr:
            legs = '  '.join(f"{g['label']} {g['ret_pct']:+.0f}%" for g in tr.get('legs', []))
            L += ['**12-month price trend.**', '',
                  f"`{tr['sparkline']}`  (weekly closes, {_fmt_pct(tr['ret_12m_pct'], dp=0)} "
                  f"over the year)", '',
                  f"- 52w high ${tr['high']:.2f} on {tr['high_date']} "
                  f"({_fmt_pct(tr['from_high_pct'], dp=0)} from here)",
                  f"- 52w low ${tr['low']:.2f} on {tr['low_date']} "
                  f"({_fmt_pct(tr['from_low_pct'], dp=0)} from here)"
                  + ('  ← low is within the last 90 days, so the fall may not be over'
                     if tr.get('bottom_is_recent') else ''),
                  f"- Quarterly legs: {legs}" if legs else '', '']

        if lv:
            L += ['**Reference levels.** Not forecasts — three independent anchors, '
                  'shown together because they disagree.', '']
            if 'analyst_mean' in lv:
                rng = ''
                if lv.get('analyst_low') and lv.get('analyst_high'):
                    rng = (f", range ${float(lv['analyst_low']):.0f}–"
                           f"${float(lv['analyst_high']):.0f}")
                L.append(f"- Analyst consensus ${lv['analyst_mean']:.2f} "
                         f"({_fmt_pct(lv['upside_analyst_pct'], dp=0)}"
                         f"{rng}, n={lv.get('analyst_n', '?')}). Sell-side targets "
                         "chase price and skew high.")
            if 'peak_3y' in lv:
                L.append(f"- Prior 3y peak ${lv['peak_3y']:.2f} "
                         f"({_fmt_pct(lv['upside_to_peak_pct'], dp=0)}); halfway back "
                         f"${lv['halfway_to_peak']:.2f} "
                         f"({_fmt_pct(lv['upside_halfway_pct'], dp=0)}). Full recovery "
                         "assumes the old multiple was deserved.")
            if 'peer_implied' in lv:
                L.append(f"- At the peer median P/S of {lv['peer_ps_median']:.1f}× "
                         f"(this name: {lv['own_ps']:.1f}×) the price would be "
                         f"${lv['peer_implied']:.2f} "
                         f"({_fmt_pct(lv['upside_peer_pct'], dp=0)}). Assumes it deserves "
                         "peer valuation, which is the thing in question.")
            L.append('')

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

        if ep and ep.get('n_peers', 0) >= 3:
            def _n(v, dp=2, pre='', suf=''):
                return (f"{pre}{float(v):.{dp}f}{suf}"
                        if isinstance(v, (int, float)) and v is not None
                        and np.isfinite(v) else 'n/a')

            L += [f"**EPS vs industry** ({ep['n_peers']} {ic.get('industry', 'industry')} "
                  f"peers with data):", '',
                  '| | This name | Industry median |', '|---|---|---|',
                  f"| EPS (TTM) | {_n(ep.get('trailing_eps'), 2, '$')} | — |",
                  f"| EPS (forward) | {_n(ep.get('forward_eps'), 2, '$')} | — |",
                  f"| EPS growth (YoY) | {_fmt_pct(ep.get('eps_growth_pct', np.nan), dp=0)} | "
                  f"{_fmt_pct(ep.get('peer_eps_growth_pct', np.nan), dp=0)} |",
                  f"| Trailing P/E | {_n(ep.get('trailing_pe'), 1, suf='×')} | "
                  f"{_n(ep.get('peer_trailing_pe'), 1, suf='×')} |",
                  f"| Forward P/E | {_n(ep.get('forward_pe'), 1, suf='×')} | "
                  f"{_n(ep.get('peer_forward_pe'), 1, suf='×')} |", '']

            notes = []
            if 'eps_growth_vs_peers_pp' in ep:
                notes.append(
                    f"EPS growth is {_fmt_pp(ep['eps_growth_vs_peers_pp'], dp=0)} vs the peer "
                    f"median ({_ord(ep['eps_growth_percentile'])} percentile) — "
                    + ("earnings are holding up better than the price implies, which is the "
                       "setup this funnel is looking for."
                       if ep['eps_growth_vs_peers_pp'] > 0 else
                       "earnings are lagging peers too, so the discount may be deserved."))
            if 'fpe_discount_pct' in ep:
                d = ep['fpe_discount_pct']
                notes.append(
                    f"Forward P/E is {abs(d):.0f}% {'below' if d < 0 else 'above'} the peer "
                    f"median. " + ("Cheap on forward earnings — but check whether those "
                                   "estimates have already been cut."
                                   if d < 0 else
                                   "Not cheap on forward earnings despite the drawdown, which "
                                   "usually means estimates fell faster than the price."))
            L += [f"- {n}" for n in notes] + ([''] if notes else [])

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
    ap.add_argument('--refresh-profiles', action='store_true',
                    help='Refetch company profiles even if cached and recent')
    ap.add_argument('--out', default=None, help='Output markdown path')
    args = ap.parse_args()

    asof, entries, latest = build(asof=args.asof, top=args.top, with_news=not args.no_news,
                                  refresh_profiles=args.refresh_profiles)
    if not entries:
        print("no candidates")
        return

    stamp = asof.strftime('%Y%m%d')
    md_path = Path(args.out) if args.out else DATA / f'dossier_{stamp}.md'
    json_path = DATA / f'dossier_{stamp}.json'
    md_path.write_text(render(asof, entries, latest))
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
