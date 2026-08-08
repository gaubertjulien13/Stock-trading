# Two-stage recommendation funnel — locked design

Status: **approved for implementation.** This is a recommendation / watchlist tool,
not a trading signal. You make every buy/sell call.

Derived from the Intel 2025 trade and the PLATFORM/CRISIS analogs in `ANALOGS.md`.
Prior research showed the mechanical filter alone has **no portfolio edge**; it is
used here only as a scanner that puts the right names on your desk.

## Stage 1 — Mechanical filter (weekly)

A name enters the candidate pool when **all** of these are true on Friday's close
(or the latest available session):

| Rule | Threshold | Why |
|---|---|---|
| Index member (S&P 500) | point-in-time | Liquidity + survivability floor |
| Price | ≥ $5 | Avoid junk |
| Median dollar volume (60d) | ≥ $20M | Tradeable |
| Price history | ≥ 400 trading days | Seasoned |
| Drawdown from 3y peak | ≤ −40% | Damage |
| Peak age | ≥ 126 trading days | Stale high, not a fresh pullback |
| Industry ETF above 200dma | true | Industry not dying |
| Industry 12m return | ≥ 0% | Tailwind |
| Stock 12m − industry 12m | ≤ −25pp | Laggard / divergence |

Optional secondary tags (not required to enter the pool):

- **CYCLE**: industry is XLE / OIH / XLB / ITB / SMH with deep cyclical history —
  labelled separately so you don't confuse it with PLATFORM/CRISIS.
- **Deep damage**: dd ≤ −60%.
- **Turning**: `rel_3m > 0` or price above 50dma — soft timing hint only.

## Stage 2 — Scored checklist (0–100)

Each Stage-1 survivor is scored on six pillars. Auto-filled where data exists;
left as `?` / manual when it requires judgment. **Your overrides always win.**

| Pillar | Weight | Auto inputs | Manual judgment |
|---|---|---|---|
| 1. Industry structural growth | 20 | industry 12m/3m return, ETF above 200dma | Is this multi-year demand or a bounce? |
| 2. Franchise / strategic asset | 20 | size (still S&P 500), dollar volume rank | Brand, IP, installed base, national role? |
| 3. Damage is temporary | 20 | peak age, rel_12m depth | Fixable miss vs secular decline? |
| 4. Concrete recovery mechanism | 15 | recent headlines (keywords: turnaround, restructuring, new CEO, CHIPS, AI, …) | Is there a real plan, or hope? |
| 5. Outside recognition | 10 | headline count/tone supporting the thesis | Are others making the same case? |
| 6. Survivability | 15 | dollar volume, vol_ann cap, recent insider buys (soft bonus) | Balance sheet / cash runway / strategic backstop? |

Score bands for the watchlist (tuned after the 2026-07-31 STRONG walk):

| Rule | Label |
|---|---|
| score ≥ 70, dd ≤ −45%, not CYCLE, not already +40% off 52w low, and (if news fetched) catalyst ≥ 10 | **STRONG** |
| score ≥ 68, catalyst ≥ 12, dd ≤ −50%, same caps | **STRONG** (activist / restructuring near-miss) |
| score ≥ 50 but fails a STRONG cap | **WATCH** |
| score < 50 | **WEAK** |

Hard caps that block STRONG: CYCLE industry tag, drawdown shallower than −45%,
already ≥ +40% off the 52-week low, or (with news) no real catalyst in headlines.

The algorithm **never** emits BUY. It emits a ranked watchlist with pillar scores,
key metrics, and the latest headline so you can accept, reject, or investigate.

## What this deliberately does not do

- No automatic position sizing or entries.
- No stop-loss / exit engine (research showed stops hurt this setup).
- No claim of backtested excess return on the Stage-2 score (the qualitative pillars
  are only partially automatable; validate forward by journaling your decisions).

## Stage-2 score validation — tested 2026-08-08, the score does not rank returns

`research/validate_stage2.py` closes the open question above. Monthly as-of dates
2007–2026, point-in-time membership, 2,288 candidate-months, 327 unique names.
Pillars 4–5 held at their production placeholders (5.0 / 3.0), since headlines do not
exist historically — so this measures the **mechanical** score, the part that runs
unattended.

**1. The score has no rank information.** Spearman IC of score vs forward return,
computed per date: **+0.017 at 3m, +0.009 at 6m, −0.030 at 12m**, all with |t| < 1.
For reference, PLAYBOOK found 0.01–0.02 for `dd_3y` and `rel_12m` and called them noise.

**2. Score quintiles slope the wrong way.** Q5-minus-Q1 forward return is
**−4.1pp at 3m, −6.0pp at 6m, −13.3pp at 12m**. Higher score, lower return.

**3. STRONG is inside the random-draw range** at every horizon (35th, 63rd, 89th
percentile against same-size random draws from the same day's candidate pool). Never
close to significant.

**4. STRONG's mean is three lottery tickets.** Over 71 STRONG candidate-months
(only **28 unique names** in 19 years), the mean 12m return is +15.62% but the
**median is −2.88% and 54% lose money**. The top three observations returned +155%,
+168% and +212%; excluding them the mean falls to **+8.44% against SPY's +16.32%**.
On non-overlapping January-only as-of dates, STRONG returns −14.3% mean / −35.2%
median, trailing SPY by 34pp on a sample of 5.

**Do not read this as "buy WEAK".** WEAK looks excellent in the overlapping sample
(+36.9% at 12m) but 23% of its observations fall in 2020, and on the non-overlapping
January-only cut it also trails SPY by 11.4pp. Overlapping monthly windows inflate
every band; the robust statement is that **no band beats the index and the ordering
carries no information**.

### What this means in practice

The score should not influence sizing or conviction. The funnel remains usable as an
idea generator — it surfaces damaged names in healthy industries for human review —
but STRONG vs WATCH vs WEAK is not a quality ranking, and the pillar total is not
evidence of anything.

It also explains why the funnel *feels* like it works. A band whose median pick loses
money while rare names return +150% to +200% produces exactly the INTC experience:
one memorable winner that reads as skill. Judge it by the median outcome, not the
one you remember.

```bash
venv/bin/python3 research/validate_stage2.py
```

## Stage 3 — the research loop (added 2026-08-08)

The validation above and `ANALOGS.md` agree from opposite directions: mechanical
features get you into the right room but do not pick the chair. Pillars 4 and 5 —
a concrete recovery mechanism and outside recognition — are the ones the analog
study credits and the only ones never tested, because they need information a price
panel does not contain. Stage 3 gathers that information so the human decision is
made against evidence rather than a score.

```
recommend.py / stage1   ->  candidates
        dossier.py      ->  fundamentals + insiders + news + peers, per candidate
        you             ->  read pillars 4-5, decide
        journal.py      ->  log decision + thesis, score it later vs SPY and vs the pool
```

**`fetch_fundamentals.py`** — point-in-time fundamentals from the SEC XBRL company
facts API. Free, no key. Every fact carries a `filed` date, so unlike yfinance this
does not leak the future and can eventually be backtested. Handles the two things
that quietly corrupt XBRL work: filers who report **cumulative year-to-date** figures
(de-cumulated by differencing entries that share a fiscal-year start) and filers who
omit standard tags (gross profit derived from cost of revenue, operating income from
gross profit minus SG&A and R&D). Restatements are resolved to the first filing, so
the series reflects what was knowable at the time.

**`dossier.py`** — one brief per candidate: setup, industry with a real peer
comparison (is the damage company-specific or sector-wide?), financial trend,
survivability, insider activity, recent headlines, and the six-pillar checklist with
pillars 4 and 5 deliberately left open. Ranked by divergence from industry rather
than by the Stage-2 score, since the score was shown to carry no information.

**`journal.py`** — records the decision *and the thesis* before the outcome is known,
then scores decisions against SPY over the intended horizon and compares buys against
passes. Since pillars 4–5 cannot be backtested, forward journaling is the only
validation route available. Roughly 20–30 decisions before anything is meaningful.

```bash
venv/bin/python3 research/dossier.py --top 12
venv/bin/python3 research/journal.py add --ticker BSX --decision buy \
    --thesis "..." --mechanism "..." --conviction 4
venv/bin/python3 research/journal.py review
```

Set a contact address once, since SEC blocks generic user agents:
`export SEC_USER_AGENT="Your Name your@email.com"` (falls back to `ALERT_FROM_EMAIL`).

### Sizing, given the shape of the payoff

`validate_stage2.py` measured the outcome distribution for this pool: the median
candidate loses money while a few return +150% to +200%. That is a venture-shaped
payoff, and the right response to it is **many small positions held long**, not
concentration into the highest-conviction name. Concentration is the intuitive move
after a win like INTC and it is the wrong one — the same distribution that produced
that win produces a majority of losers.

## Outputs

- Console ranked table
- `research/data/watchlist_YYYYMMDD.csv` — full metrics + pillar scores
- `research/data/watchlist_YYYYMMDD.md` — human-readable brief for review

## Reproduce / run

```
venv/bin/python3 research/recommend.py              # scan as-of latest cached date
venv/bin/python3 research/recommend.py --asof 2024-05-10   # historical freeze (INTC era)
venv/bin/python3 research/recommend.py --news               # fetch yfinance headlines for top names
venv/bin/python3 research/recommend.py --news --email       # email STRONG digest (uses .stock_screener.env)
venv/bin/python3 research/recommend.py --news --email --email-on-new   # only if new STRONG names
```

Weekly cron example (Sunday evening Pacific):

```
0 18 * * 0 cd /path/to/finance && venv/bin/python3 research/recommend.py --news --email
```
