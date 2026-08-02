"""Two-stage Intel-style recommendation funnel.

Stage 1: mechanical filter (damaged laggard in a healthy industry).
Stage 2: scored checklist -> ranked watchlist.
Optional: email digest of STRONG names (same SMTP env as the daily alert script).

This does NOT emit buy signals. It puts candidates on your desk with pillar scores
so you make the call the way you did on Intel.

  venv/bin/python3 research/recommend.py
  venv/bin/python3 research/recommend.py --asof 2024-05-10
  venv/bin/python3 research/recommend.py --news --top 25
  venv/bin/python3 research/recommend.py --news --email
  venv/bin/python3 research/recommend.py --news --email --email-on-new
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA

# Same credentials file as Trading_Buy_Signal_Strict_Script_With_Alerts.py
load_dotenv(Path(__file__).resolve().parents[1] / '.stock_screener.env')

ALERT_STATE = DATA / 'recommend_alert_state.json'
ALERT_LOG = DATA / 'recommend_alert_log.csv'

pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 60)

CYCLE_INDUSTRIES = {'XLE', 'OIH', 'XLB', 'ITB', 'KRE'}

# Strong cues: explicit turnaround / fix language (PLATFORM/CRISIS analogs).
CATALYST_STRONG = re.compile(
    r'\b(turnaround|restructuring|restructur|new ceo|chief executive|'
    r'cost.?cut|year of efficiency|buyback|foundry|chips act|'
    r'strategic review|spin.?off|activist|overhaul|relaunch|'
    r'recovery plan|path to profitability)\b',
    re.I,
)
# Weaker cues: demand/product backdrop — common in bull markets, so capped.
CATALYST_SOFT = re.compile(
    r'\b(pipeline|partnership|contract win|data center|hyperscale|'
    r'approval|undervalued|oversold|rebound)\b',
    re.I,
)
THESIS_KW = re.compile(
    r'\b(cheap|bargain|opportunity|mispriced|turnaround|recovery|'
    r'undervalued|beaten.?down|oversold|buy the dip)\b',
    re.I,
)
HARD_NEG_KW = re.compile(
    r'\b(bankrupt|chapter 11|going concern|fraud|restatement|'
    r'delisting|default|liquidation|criminal)\b',
    re.I,
)


# --------------------------------------------------------------------------- Stage 1

def stage1_mask(f, d, asof):
    close = d['close']
    member = d['member'].reindex(columns=close.columns).fillna(False)
    liquid = (f['dollar_vol'] >= 20e6) & (close >= 5.0)
    damaged = (f['dd_3y'] <= -0.40) & (f['peak_age_days'] >= 126)
    ind_ok = f['ind_above_200'].fillna(False) & (f['ind_ret_12m'] >= 0)
    diverge = f['rel_12m'] <= -25.0
    seasoned = f['age_days'] >= 400
    valid = close.notna()

    mask = member & liquid & damaged & ind_ok & diverge & seasoned & valid
    row = mask.loc[asof]
    return row[row].index.tolist()


def snapshot(ticker, asof, f, d):
    c = d['close'].at[asof, ticker]
    return dict(
        ticker=ticker,
        asof=asof.date().isoformat(),
        price=float(c),
        dd_3y_pct=float(f['dd_3y'].at[asof, ticker] * 100),
        peak_age_days=float(f['peak_age_days'].at[asof, ticker]),
        ret_12m=float(f['ret_12m'].at[asof, ticker]),
        ret_3m=float(f['ret_3m'].at[asof, ticker]),
        ret_1m=float(f['ret_1m'].at[asof, ticker]),
        rel_12m=float(f['rel_12m'].at[asof, ticker]),
        rel_3m=float(f['rel_3m'].at[asof, ticker]),
        ind_ret_12m=float(f['ind_ret_12m'].at[asof, ticker]),
        ind_ret_3m=float(f['ind_ret_3m'].at[asof, ticker]),
        industry=str(f['industry'].at[asof, ticker]),
        dollar_vol_m=float(f['dollar_vol'].at[asof, ticker] / 1e6),
        vol_ann=float(f['vol_ann'].at[asof, ticker]),
        off_52w_low_pct=float(f['off_52w_low_pct'].at[asof, ticker]),
        dist_200_pct=float(f['dist_200_pct'].at[asof, ticker]),
        above_200=bool(f['above_200'].at[asof, ticker]),
        above_50=bool(f['above_50'].at[asof, ticker]),
        sma200_slope=float(f['sma200_slope'].at[asof, ticker]),
        spy_above_200=bool(f['spy_above_200'].at[asof, ticker]),
    )


# --------------------------------------------------------------------------- Stage 2

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def score_industry(s):
    """Pillar 1 — structural growth proxy from industry ETF path (0-20)."""
    pts = 0.0
    if s['ind_ret_12m'] >= 20:
        pts += 10
    elif s['ind_ret_12m'] >= 10:
        pts += 7
    elif s['ind_ret_12m'] >= 0:
        pts += 4
    if s['ind_ret_3m'] >= 5:
        pts += 5
    elif s['ind_ret_3m'] >= 0:
        pts += 3
    # SMH / XLK / IBB / XLY often house PLATFORM analogs; soft boost
    if s['industry'] in {'SMH', 'XLK', 'IBB', 'XLC'}:
        pts += 5
    elif s['industry'] in {'XLY', 'XLI', 'XLV'}:
        pts += 3
    return _clip(pts, 0, 20)


def score_franchise(s, dv_percentile):
    """Pillar 2 — franchise proxy: size/liquidity rank inside the candidate set (0-20)."""
    pts = 8.0  # base: still an S&P 500 name that cleared Stage 1
    pts += 12 * dv_percentile  # larger/more liquid -> more franchise-like
    if s['price'] >= 20:
        pts += 2
    return _clip(pts, 0, 20)


def score_temporary(s):
    """Pillar 3 — damage looks temporary vs already healing (0-20).

    Deep, stale damage with industry intact scores high. Signs the decline is
    still accelerating (rel_3m << 0 and new lows) score lower — more knife.
    """
    pts = 0.0
    dd = abs(s['dd_3y_pct'])
    if dd >= 60:
        pts += 8
    elif dd >= 50:
        pts += 6
    else:
        pts += 4
    if s['peak_age_days'] >= 252:
        pts += 4
    elif s['peak_age_days'] >= 126:
        pts += 2
    # Divergence depth
    if s['rel_12m'] <= -50:
        pts += 5
    elif s['rel_12m'] <= -35:
        pts += 3
    else:
        pts += 1
    # Soft turn: no longer making relative lows
    if s['rel_3m'] > 0:
        pts += 3
    elif s['rel_3m'] > -10:
        pts += 1
    # Still near the lows looks more Intel-like than a name already +60% off the bottom
    if s['off_52w_low_pct'] <= 15:
        pts += 2
    elif s['off_52w_low_pct'] >= 50:
        pts -= 4
    return _clip(pts, 0, 20)


def score_survivability(s, insider_bonus=0.0):
    """Pillar 6 — can it live long enough to turn (0-15)."""
    pts = 0.0
    if s['dollar_vol_m'] >= 100:
        pts += 6
    elif s['dollar_vol_m'] >= 50:
        pts += 4
    else:
        pts += 2
    if s['vol_ann'] <= 40:
        pts += 5
    elif s['vol_ann'] <= 60:
        pts += 3
    elif s['vol_ann'] <= 80:
        pts += 1
    # Extreme vol in a damaged name is often distress, not opportunity
    if s['vol_ann'] > 100:
        pts -= 3
    pts += insider_bonus  # 0-4 from Form 4 cluster
    return _clip(pts, 0, 15)


def score_catalyst_from_news(headlines):
    """Pillar 4 — concrete recovery mechanism cues in headlines (0-15)."""
    if not headlines:
        return 5.0, '?', ''  # neutral placeholder when news not fetched
    strong = soft = neg = 0
    best = ''
    for h in headlines:
        if HARD_NEG_KW.search(h):
            neg += 1
        if CATALYST_STRONG.search(h):
            strong += 1
            if not best:
                best = h
        elif CATALYST_SOFT.search(h):
            soft += 1
            if not best:
                best = h
    if neg:
        return 0.0, 'hard-negative', headlines[0][:120]
    if strong >= 2:
        return 15.0, 'strong', best[:120]
    if strong >= 1:
        return 12.0, 'present', best[:120]
    if soft >= 2:
        return 7.0, 'soft', best[:120]
    if soft >= 1:
        return 5.0, 'soft', best[:120]
    return 3.0, 'none-seen', (headlines[0][:120] if headlines else '')


def score_recognition_from_news(headlines):
    """Pillar 5 — outside recognition of the thesis (0-10)."""
    if not headlines:
        return 3.0, '?'
    n = sum(1 for h in headlines
            if THESIS_KW.search(h) or CATALYST_STRONG.search(h))
    if n >= 3:
        return 10.0, 'widely discussed'
    if n >= 1:
        return 6.0, 'mentioned'
    return 2.0, 'quiet'


def insider_bonus(ticker, asof, buys):
    """0-4 points if open-market insider buying clustered in the prior 90 days."""
    if buys is None or buys.empty:
        return 0.0
    lo = asof - pd.Timedelta(days=90)
    sub = buys[(buys.ticker == ticker) &
               (buys.filing_date >= lo) &
               (buys.filing_date <= asof)]
    if sub.empty:
        return 0.0
    n = int(sub['n_insiders'].sum())
    if n >= 5:
        return 4.0
    if n >= 3:
        return 3.0
    if n >= 1:
        return 1.5
    return 0.0


def fetch_headlines(ticker, max_items=10):
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items[:max_items]:
        content = it.get('content') or {}
        title = (content.get('title') or it.get('title') or '').strip()
        if title:
            out.append(title)
    return out


def stage2(candidates, asof, f, d, buys=None, fetch_news=False, news_top=15):
    snaps = [snapshot(t, asof, f, d) for t in candidates]
    df = pd.DataFrame(snaps)
    if df.empty:
        return df

    # dollar-volume percentile within the candidate set (franchise proxy)
    df['dv_pct'] = df['dollar_vol_m'].rank(pct=True)

    rows = []
    # Pre-fetch news only for the most damaged / divergent names to limit API calls
    news_order = df.sort_values(['rel_12m', 'dd_3y_pct']).ticker.tolist()
    news_set = set(news_order[:news_top]) if fetch_news else set()

    for _, s in df.iterrows():
        s = s.to_dict()
        ib = insider_bonus(s['ticker'], asof, buys)
        headlines = fetch_headlines(s['ticker']) if s['ticker'] in news_set else []

        p1 = score_industry(s)
        p2 = score_franchise(s, s['dv_pct'])
        p3 = score_temporary(s)
        p4, p4_label, headline = score_catalyst_from_news(headlines)
        p5, p5_label = score_recognition_from_news(headlines)
        p6 = score_survivability(s, ib)

        # If news wasn't fetched, keep pillars 4/5 as mid placeholders and flag them
        news_status = 'fetched' if s['ticker'] in news_set else 'not-fetched'
        if news_status == 'not-fetched':
            p4, p4_label = 5.0, 'manual'
            p5, p5_label = 3.0, 'manual'
            headline = ''

        total = p1 + p2 + p3 + p4 + p5 + p6

        tag = 'CYCLE' if s['industry'] in CYCLE_INDUSTRIES else 'PLATFORM/CRISIS'
        if s['dd_3y_pct'] <= -60:
            tag += '|DEEP'
        if s['rel_3m'] > 0 or s['above_50']:
            tag += '|TURNING'

        band = assign_band(
            score=total, dd=s['dd_3y_pct'], p4=p4, tag=tag,
            news_status=news_status, off_low=s['off_52w_low_pct'],
        )

        rows.append({
            **{k: s[k] for k in (
                'ticker', 'asof', 'price', 'dd_3y_pct', 'rel_12m', 'rel_3m',
                'ind_ret_12m', 'industry', 'dollar_vol_m', 'vol_ann',
                'off_52w_low_pct', 'dist_200_pct', 'above_50', 'above_200')},
            'tag': tag,
            'p1_industry': round(p1, 1),
            'p2_franchise': round(p2, 1),
            'p3_temporary': round(p3, 1),
            'p4_catalyst': round(p4, 1),
            'p4_label': p4_label,
            'p5_recognition': round(p5, 1),
            'p5_label': p5_label,
            'p6_survivability': round(p6, 1),
            'insider_bonus': ib,
            'score': round(total, 1),
            'band': band,
            'headline': headline,
            'news_status': news_status,
            'manual_override': '',  # fill in by hand
            'thesis_notes': '',
        })

    out = pd.DataFrame(rows).sort_values(
        ['score', 'rel_12m'], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out


def assign_band(score, dd, p4, tag, news_status, off_low):
    """Tuned banding from the 2026-07-31 STRONG walk.

    STRONG must look like a PLATFORM/CRISIS analog, not a quality pullback,
    a crypto/cycle washout, or a name with no recovery mechanism in the tape.
    """
    if score < 50:
        return 'WEAK'

    # Hard caps — these never get STRONG automatically
    if tag.startswith('CYCLE'):
        return 'WATCH'
    if dd > -45:          # shallower than -45%: quality pullback, not Intel-shape
        return 'WATCH'
    if off_low >= 40:     # already bounced hard off the 52w low
        return 'WATCH'
    # When news was fetched, demand a real catalyst for STRONG
    if news_status == 'fetched' and p4 < 10:
        return 'WATCH'

    # Primary bar
    if score >= 70:
        return 'STRONG'
    # Activist / restructuring near-miss (e.g. FISV): deep damage + clear catalyst
    if score >= 68 and p4 >= 12 and dd <= -50:
        return 'STRONG'
    return 'WATCH'


# --------------------------------------------------------------------------- report

def write_markdown(df, path, asof):
    lines = [
        f'# Intel-style watchlist — {asof.date()}',
        '',
        'Recommendation only. No automatic buys. Pillars 4–5 need your judgment '
        'when marked `manual`.',
        '',
        f'Candidates after Stage 1: **{len(df)}**  |  '
        f'STRONG: **{(df.band == "STRONG").sum()}**  |  '
        f'WATCH: **{(df.band == "WATCH").sum()}**  |  '
        f'WEAK: **{(df.band == "WEAK").sum()}**',
        '',
        '| Rank | Ticker | Score | Band | Tag | Price | DD 3y | Rel 12m | Industry | Ind 12m | Notes |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r.ticker} | {r.score:.0f} | {r.band} | {r.tag} | "
            f"${r.price:.1f} | {r.dd_3y_pct:.0f}% | {r.rel_12m:.0f}pp | {r.industry} | "
            f"{r.ind_ret_12m:.0f}% | {r.headline[:50] if r.headline else r.p4_label} |"
        )
    lines += [
        '',
        '## Pillar key',
        '',
        '| Code | Pillar | Max |',
        '|---|---|---|',
        '| p1 | Industry structural growth | 20 |',
        '| p2 | Franchise / strategic asset | 20 |',
        '| p3 | Damage looks temporary | 20 |',
        '| p4 | Concrete recovery mechanism | 15 |',
        '| p5 | Outside recognition | 10 |',
        '| p6 | Survivability | 15 |',
        '',
        '## How to use',
        '',
        '1. Start with STRONG, then WATCH.',
        '2. For each name, fill `thesis_notes` and override pillar scores if the '
        'auto score disagrees with your read.',
        '3. Reject anything where pillar 3 fails on judgment (secular decline) '
        'even if the total score is high.',
        '4. Journal the decision — that is how we validate Stage 2 forward.',
        '',
    ]
    # Detail blocks for top 10
    lines.append('## Top candidates — pillar detail')
    lines.append('')
    for _, r in df.head(10).iterrows():
        lines += [
            f"### {r.ticker}  —  {r.score:.0f}/100  ({r.band})",
            '',
            f"- Price ${r.price:.2f} | drawdown {r.dd_3y_pct:.1f}% | "
            f"rel 12m {r.rel_12m:.1f}pp | {r.industry} +{r.ind_ret_12m:.1f}%",
            f"- p1 industry {r.p1_industry} | p2 franchise {r.p2_franchise} | "
            f"p3 temporary {r.p3_temporary} | p4 catalyst {r.p4_catalyst} ({r.p4_label}) | "
            f"p5 recognition {r.p5_recognition} ({r.p5_label}) | "
            f"p6 survivability {r.p6_survivability}",
            f"- Tags: {r.tag}",
        ]
        if r.headline:
            lines.append(f"- Headline: {r.headline}")
        lines.append('')
    path.write_text('\n'.join(lines))


# --------------------------------------------------------------------------- email

def send_email(smtp_host, smtp_port, smtp_user, smtp_pass, to_list, subject, body):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = smtp_user or 'alerts@localhost'
    msg['To'] = ', '.join(to_list)
    msg.set_content(body)
    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            try:
                s.starttls()
                s.ehlo()
            except Exception:
                pass
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)


def load_alert_state():
    if ALERT_STATE.exists():
        return json.loads(ALERT_STATE.read_text())
    return {'last_strong': [], 'last_sent': None}


def save_alert_state(state):
    ALERT_STATE.write_text(json.dumps(state, indent=1))


def log_recommend_alert(asof, strong, new_tickers):
    file_exists = ALERT_LOG.exists() and ALERT_LOG.stat().st_size > 0
    with open(ALERT_LOG, 'a', newline='') as fh:
        w = csv.writer(fh)
        if not file_exists:
            w.writerow(['timestamp', 'asof', 'ticker', 'score', 'band', 'tag',
                        'dd_3y_pct', 'rel_12m', 'industry', 'is_new', 'headline'])
        now = datetime.now().isoformat(timespec='seconds')
        new_set = set(new_tickers)
        for _, r in strong.iterrows():
            w.writerow([now, asof.date().isoformat(), r.ticker, r.score, r.band, r.tag,
                        round(r.dd_3y_pct, 1), round(r.rel_12m, 1), r.industry,
                        r.ticker in new_set, (r.headline or '')[:140]])


def build_digest(strong, watch_near, asof, new_tickers):
    lines = [
        f'Intel-style watchlist digest — {asof.date()}',
        '',
        'Recommendation only. You make every trade call.',
        f'STRONG names: {len(strong)}'
        + (f'  |  NEW since last email: {", ".join(new_tickers)}' if new_tickers else ''),
        '',
    ]
    if strong.empty:
        lines += ['No STRONG candidates after tuned banding.', '']
    else:
        lines.append('=== STRONG ===')
        lines.append('')
        for _, r in strong.iterrows():
            flag = '  [NEW]' if r.ticker in new_tickers else ''
            lines += [
                f"{r.ticker}  {r.score:.0f}/100{flag}",
                f"  ${r.price:.2f}  |  dd {r.dd_3y_pct:.0f}%  |  rel12 {r.rel_12m:.0f}pp  |  "
                f"{r.industry} +{r.ind_ret_12m:.0f}%  |  {r.tag}",
                f"  pillars  ind {r.p1_industry}  fran {r.p2_franchise}  temp {r.p3_temporary}  "
                f"cat {r.p4_catalyst} ({r.p4_label})  rec {r.p5_recognition}  "
                f"surv {r.p6_survivability}",
            ]
            if r.headline:
                lines.append(f"  headline: {r.headline[:140]}")
            lines.append('')

    if not watch_near.empty:
        lines.append('=== Near-miss WATCH (score >= 65) — worth a glance ===')
        lines.append('')
        for _, r in watch_near.iterrows():
            lines.append(
                f"{r.ticker:5}  {r.score:.0f}  dd {r.dd_3y_pct:.0f}%  "
                f"rel12 {r.rel_12m:.0f}pp  cat {r.p4_catalyst} ({r.p4_label})  {r.tag}"
            )
            if r.headline:
                lines.append(f"       {r.headline[:100]}")
        lines.append('')

    lines += [
        'Reject if the damage looks secular (cord-cutting, obsolete product).',
        'Journal decisions — that is how Stage 2 gets validated forward.',
        '',
        f'Full watchlist: research/data/watchlist_{asof.strftime("%Y%m%d")}.md',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asof', default=None, help='YYYY-MM-DD (default: latest cached)')
    ap.add_argument('--news', action='store_true',
                    help='Fetch yfinance headlines for top damaged names (live)')
    ap.add_argument('--top', type=int, default=25,
                    help='How many names to enrich with news when --news is set')
    ap.add_argument('--min-band', choices=['STRONG', 'WATCH', 'WEAK'], default='WEAK',
                    help='Lowest band to include in the printed table')
    ap.add_argument('--email', action='store_true',
                    help='Email the STRONG digest using .stock_screener.env')
    ap.add_argument('--email-on-new', action='store_true',
                    help='With --email, only send if a new ticker entered STRONG')
    ap.add_argument('--smtp-host', default='smtp.gmail.com')
    ap.add_argument('--smtp-port', type=int, default=465)
    ap.add_argument('--smtp-user', default=None)
    ap.add_argument('--smtp-pass-env', default='SMTP_APP_PASSWORD')
    ap.add_argument('--email-to', default=None,
                    help='Comma-separated; default ALERT_TO_EMAILS from env')
    args = ap.parse_args()

    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)

    if args.asof:
        asof = pd.Timestamp(args.asof)
        if asof not in d['close'].index:
            asof = d['close'].index[d['close'].index.get_indexer([asof], method='ffill')[0]]
    else:
        asof = d['close'].index[-1]

    buys = None
    buy_path = DATA / 'insider_buys.parquet'
    if buy_path.exists():
        buys = pd.read_parquet(buy_path)
        buys['filing_date'] = pd.to_datetime(buys['filing_date'])

    tickers = stage1_mask(f, d, asof)
    print('=' * 100)
    print(f'STAGE 1  —  Intel-shape filter as of {asof.date()}')
    print('=' * 100)
    print(f'Candidates: {len(tickers)}')
    if not tickers:
        print('No names passed the filter.')
        return

    # Email runs always fetch news so catalyst gating for STRONG is meaningful
    fetch_news = args.news or args.email
    print('Scoring Stage 2...', flush=True)
    wl = stage2(tickers, asof, f, d, buys=buys, fetch_news=fetch_news, news_top=args.top)

    band_order = {'STRONG': 0, 'WATCH': 1, 'WEAK': 2}
    keep = {b for b, i in band_order.items() if i <= band_order[args.min_band]}
    shown = wl[wl.band.isin(keep)]

    cols = ['rank', 'ticker', 'score', 'band', 'tag', 'price', 'dd_3y_pct', 'rel_12m',
            'industry', 'ind_ret_12m', 'p1_industry', 'p2_franchise', 'p3_temporary',
            'p4_catalyst', 'p5_recognition', 'p6_survivability']
    print()
    print('=' * 100)
    print('STAGE 2  —  ranked watchlist (tuned banding)')
    print('=' * 100)
    print(shown[cols].to_string(index=False))

    stamp = asof.strftime('%Y%m%d')
    csv_path = DATA / f'watchlist_{stamp}.csv'
    md_path = DATA / f'watchlist_{stamp}.md'
    wl.to_csv(csv_path, index=False)
    write_markdown(wl, md_path, asof)
    print()
    print(f'Wrote {csv_path}')
    print(f'Wrote {md_path}')
    if not fetch_news:
        print('Tip: re-run with --news to auto-score catalyst/recognition from headlines.')

    strong = wl[wl.band == 'STRONG'].copy()
    watch_near = wl[(wl.band == 'WATCH') & (wl.score >= 65)].copy()

    state = load_alert_state()
    prev = set(state.get('last_strong') or [])
    now_set = set(strong.ticker.tolist())
    new_tickers = sorted(now_set - prev)

    if args.email:
        smtp_user = args.smtp_user or os.environ.get('ALERT_FROM_EMAIL', '')
        to_raw = args.email_to or os.environ.get('ALERT_TO_EMAILS', '')
        to_list = [x.strip() for x in to_raw.split(',') if x.strip()]
        smtp_pass = os.environ.get(args.smtp_pass_env, '')
        if not to_list:
            print('ERROR: --email needs --email-to or ALERT_TO_EMAILS in .stock_screener.env')
            return
        if args.email_on_new and not new_tickers and state.get('last_sent'):
            print(f'No new STRONG names since {state.get("last_sent")} — email skipped.')
            # still refresh state snapshot of current STRONG
            state['last_strong'] = sorted(now_set)
            save_alert_state(state)
            return

        body = build_digest(strong, watch_near, asof, new_tickers)
        subject = (
            f'Intel watchlist {asof.date()} — '
            f'{len(strong)} STRONG'
            + (f' ({len(new_tickers)} new)' if new_tickers else '')
        )
        print(f'Sending email to {to_list} …', flush=True)
        send_email(args.smtp_host, args.smtp_port, smtp_user, smtp_pass,
                   to_list, subject, body)
        log_recommend_alert(asof, strong, new_tickers)
        state['last_strong'] = sorted(now_set)
        state['last_sent'] = datetime.now().isoformat(timespec='seconds')
        save_alert_state(state)
        print(f'Sent. Logged -> {ALERT_LOG}')
    else:
        print()
        print(f'STRONG ({len(strong)}): {", ".join(strong.ticker.tolist()) or "(none)"}')
        if new_tickers:
            print(f'New vs last emailed set: {", ".join(new_tickers)}')


if __name__ == '__main__':
    main()
