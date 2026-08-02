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
