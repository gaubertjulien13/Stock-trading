# Buy Signal System — Reference & Operating Manual

Living document for `Trading_Buy_Signal_Strict_Script_With_Alerts.py` and its
supporting tools. Covers what the system does, how to run it, what has been
validated or rejected in backtests, and the operational failure modes we've hit.

Last updated: 2026-08-03

---

## 1. Goal

A **short-swing (1–5 day) buy-signal system** for a real-money account holding
**1–5 concurrent positions**, optimized for **expectancy** (average R per trade)
rather than win rate or signal count.

Design principle: *fewer signals with genuine gain potential* beats more signals.
Most of the work in this system is in **refusing** trades, not finding them.

---

## 2. Quick start

Run before 6:30am PT in a terminal you leave open all day:

```bash
cd /Users/juliengaubert/cursor/finance
./venv/bin/python Trading_Buy_Signal_Strict_Script_With_Alerts.py \
  --cli --live \
  --interval 1h --period 60d \
  --universe sp500 \
  --score-threshold 11 \
  --smtp-user "gaubertjulien13@gmail.com" \
  --email-to "gaubertjulien13@gmail.com"
```

Confirm the startup banner shows the filters are on, especially:

```
💓 Heartbeat email: ON (one per trading day, any regime)  |  Network timeout: 30s (anti-hang)
```

Credentials come from `.stock_screener.env` (git-ignored): `SMTP_APP_PASSWORD`,
`ALERT_FROM_EMAIL`, `ALERT_TO_EMAILS`.

### Daily rhythm

| Time (PT) | What happens |
|---|---|
| 6:30am | Market opens. **Baseline scan**: pre-existing signals are logged, no emails sent, email budget untouched. |
| 6:30–7:30am | **Quiet period**. Signals accumulate in `alert_log.csv`; individual alerts held back. |
| 7:30am | **One DAILY PICKS email** — top 3 ranked, one per sector. Also the daily **heartbeat** email goes out (first scan of the day, before any suppression). |
| after 7:30am | Individual alert emails resume for *fresh* threshold crossings only, capped at 6/day and 2/sector. |
| 1:00pm | Market closes; the loop sleeps until the next session. |

**Trade from the 7:30 picks email, not the individual alerts.** The individual
alerts are first-come-first-served against the daily budget, so a 15-point signal
arriving at 11am can be crowded out by six 11-point signals from the morning. The
picks email ranks the whole pool properly.

Ad-hoc re-ranking at any point in the day (run in a *second* terminal):

```bash
./venv/bin/python pick_daily_alerts.py --max-picks 3
```

---

## 3. How the system works

### 3.1 Data

- **Daily bars** (refreshed once per calendar day) provide trend context:
  SMA200, ADX, ATR, 20-day relative strength vs SPY, and 5-day return.
- **Intraday bars** (1h, refreshed every poll) provide entry timing:
  RSI (Wilder 14), Bollinger Bands, volume, MACD histogram.
- **Sector ETFs** (the 11 SPDRs) provide sector-level 5-day returns.
- Source is `yfinance` throughout — free, and occasionally flaky (see §7).

### 3.2 Market regime kill switch

Computed daily from SPY plus market breadth:

| Regime | Trigger | Behavior |
|---|---|---|
| `risk_off` | SPY below SMA200, **or** SPY 5-day return < −5% | All alerts suppressed |
| `caution` | SPY below SMA50, **or** breadth < 40%, **or** 5-day < −2% | All alerts suppressed (`--skip-caution`, default on) |
| `risk_on` | none of the above | Normal operation |

Breadth = % of scanned tickers above their own daily SMA200.

### 3.3 Scoring (max 17 points)

| Component | Points | Condition |
|---|---|---|
| Daily trend | 3 | Above daily SMA200 |
| ADX | 2 / 1 | Strong (>35) / trending (>25) |
| Relative strength | 2 | Outperforming SPY over 20 days |
| RSI zone | 3 / 1 | 40–60 and rising ("sweet spot") / 60–70 ("warm") |
| Bollinger | 3 / 1 | Lower-band bounce / middle-band cross |
| Volume | 2 / 1 | Surge / above average |
| MACD histogram | 2 / 1 | Positive and rising / rising |

Default threshold is **11**. An alert fires when the score *crosses* the
threshold from below (not merely sits above it).

> **Do not raise the threshold to 14–15.** See §5 — high scores concentrate in
> Bollinger-bounce dip-buys, which have *lower* expectancy. This has been
> confirmed twice in live trading and in backtest.

### 3.4 Vetoes (applied to every candidate, in order)

| Veto | Default | Rule |
|---|---|---|
| Momentum band | on | Reject if stock's 5-day move is outside −6%…+6% (falling knives and over-extended chases) |
| Sector guard | on | Reject if the stock's sector ETF 5-day return ≤ −5% |
| Earnings blackout | on | Reject if earnings within 7 days |
| News veto | on | Reject on strongly negative recent headlines (VADER + keyword) |

Vetoed candidates are still written to `alert_log.csv` with the reason, so
rejections stay auditable.

### 3.5 Email tiering

Everything above threshold is **logged**; only a subset is **emailed**:

1. Must be a **mild dip** (5-day move between −6% and 0%), where backtested edge concentrates.
2. Max **6 emails per day**.
3. Max **2 emails per sector per day**.

Dry-run evidence: on 2026-07-06 this reduced 101 emails to 6.

### 3.6 Trade plan in every alert

- Entry: the alert price
- Stop: **1.0 × daily ATR** below entry
- Target: **3.0 × daily ATR** above entry
- Time stop: exit by day 5 regardless

---

## 4. CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--score-threshold` | 11 | Minimum score to trigger |
| `--interval` / `--period` | 1h / 60d | Intraday bar size and history |
| `--universe` | sp500 | `sp500` or `nasdaq` |
| `--poll-secs` | 120 | Seconds between scans |
| `--debounce-mins` | 60 | Per-ticker alert cooldown |
| `--momentum-band` / `--band-min` / `--band-max` | on / −6 / +6 | Falling-knife and chase guard |
| `--sector-guard` / `--sector-min` | on / −5 | Sector ETF crash guard |
| `--earnings-blackout` / `--earnings-blackout-days` | on / 7 | Earnings gap risk |
| `--news-veto` / `--news-lookback-hours` | on / 72 | Negative news veto |
| `--skip-caution` | on | Suppress everything in caution regime |
| `--email-mild-dip-only` | on | Email only mild dips |
| `--max-emails-per-day` | 6 | Hard daily email cap |
| `--max-emails-per-sector` | 2 | Per-sector daily email cap |
| `--daily-picks` / `--picks-time` / `--picks-count` | on / 07:30 / 3 | Daily picks email |
| `--status-email` | on | Daily heartbeat email |
| `--alert-log` | alert_log.csv | Where signals are logged |

---

## 5. Validated changes (adopted)

All validated in `backtest_harness.py` across six market-regime windows with a
train/test split, next-bar-open fills, and slippage. Baseline expectancy is
**+0.185R** per trade (out-of-sample +0.475R on the holdout).

| Change | Evidence | Status |
|---|---|---|
| 1.0×ATR stop / 3.0×ATR target | ~48% better holdout expectancy than alternatives | Adopted |
| Score threshold 11 | Best balance of expectancy vs trade count; higher thresholds overfit | Adopted |
| Earnings blackout | Tail-risk control against overnight gaps | Adopted |
| Momentum band (−6%…+6%) | Expectancy concentrates in mild dips (+0.27…+0.30R vs +0.10 elsewhere) | Adopted |
| Sector kill switch (≤ −5%) | Sector-crash trades: −0.29R, win 14%, PF 0.63 | Adopted |
| Skip caution regime entirely | Caution trades −0.11R vs risk-on +0.19R; tighter targets didn't help (−0.08R) | Adopted |
| News veto | Not backtestable (no historical headline archive) — adopted on forward-validation basis | Adopted, unproven |

---

## 6. Rejected changes — **do not retry without new evidence**

These were tested properly and failed. Documented so we don't burn time re-testing.

| Idea | Result | Why it fails |
|---|---|---|
| **15-minute bars** for faster detection | +0.142R vs **+0.368R** for 1h on the same window | Faster confirmation fires on noise; the lag is part of the edge |
| **Retrace-limit entries** (buy 0.15–0.40×ATR below the alert) | Per-signal expectancy drops from +0.185R to +0.162/+0.129/+0.095R | Adverse selection — the signals that never pull back are the best ones |
| **Top-N ranking** (take only the day's highest scorers) | Roughly halves expectancy | Concentrates into correlated, extended names |
| **Reweighting the score categories** | No out-of-sample improvement | Overfits the training windows |
| **Per-day sector caps as a signal filter** | +0.16R capped vs +0.19R uncapped | Diversification is a *portfolio* concern, not a signal one — moved into `pick_daily_alerts.py` |
| **Extra entry filters** (bb_position, atr_pct thresholds) | No out-of-sample improvement | Overfits |

---

## 7. Operational failure modes (learned the hard way)

| Symptom | Cause | Fix / status |
|---|---|---|
| Silence for days, terminal frozen | A dead yfinance/SMTP socket blocked a read forever (no default timeout). Cost 4 trading days, Jul 16–20. | **Fixed** — `socket.setdefaulttimeout(30)` at startup |
| Silence, no way to tell if the script died | Caution regime and a dead process look identical from the inbox | **Fixed** — daily heartbeat email (`--status-email`) sent before any suppression. *No heartbeat on a trading day = the script is down.* |
| Burst of alerts at launch, then nothing all day | Every signal already above threshold looked like a fresh crossing and drained the 6/day budget | **Fixed** — baseline scan on startup and on each day rollover: logs without emailing |
| Persistent overnight setups missing from picks | Score memory carried across days, so no "crossing" occurred | **Fixed** — score memory resets each trading day |
| Script dies when terminal window closes | Process is tied to the terminal | **Habit** — keep the terminal open; don't close the window |
| Sporadic `Failed downloads` / `YFRateLimitError` | yfinance flakiness | Benign — those tickers are skipped for one scan and recover |

---

## 8. Live performance history

Simulated from `alert_log.csv` against real 15m data (entry at alert price,
1×ATR stop, 3×ATR target, unresolved marked at week close).

**Week of Jul 6–10, 2026** (SPY +0.83%):

| Cohort | Trades | Avg R | Stopped | Winners |
|---|---|---|---|---|
| Everything logged | 195 | −0.13 | 38% | 42% |
| **Daily picks** | 12 | **+0.02** | 17% | 50% |
| Mild-dip email tier | 90 | −0.08 | 39% | 44% |
| Score ≥ 14 | 7 | −0.94 | 86% | 0% |
| Vetoed (avoided) | 90 | −0.20 | 42% | 38% |

Reading: the funnel ordered correctly — picks beat the email tier, which beat the
raw pool, which beat the rejects. But the underlying signal engine had a poor week
and zero trades reached the 3×ATR target. **Live edge remains unproven**; the
+0.19R backtest expectancy has not yet shown up in live results.

Regime history: no signals were logged between Jul 24 and Jul 31. The regime
check confirmed **caution on every one of those sessions** (SPY below SMA50 while
breadth stayed healthy at 66–72%), so the six-session silence was correct behavior,
not a failure. Aug 3 was risk-on: 94 signals logged, 56 non-vetoed, 6 alerts
emailed (cap reached), picks were BAC / A / HRL.

---

## 9. Files

| File | Purpose |
|---|---|
| `Trading_Buy_Signal_Strict_Script_With_Alerts.py` | The live scanner. This is the system. |
| `pick_daily_alerts.py` | Ranks a day's logged alerts (one per sector, mild dips and volume surges first, score as tiebreak). Also used internally for the picks email. |
| `backtest_harness.py` | Validation harness: multi-window, train/test split, lookahead-safe, expectancy-first. **Every change goes through this before going live.** |
| `analyze_week.py` | Replays a week of logged alerts against real 15m data to grade live performance by cohort. Update the dates at the top. |
| `check_week_regime.py` | Reconstructs the daily market regime for a date range — answers "was the silence correct?" |
| `myutils.py` | Ticker universe fetching (S&P 500, Nasdaq) with caching |
| `alert_log.csv` | Every signal ever logged, vetoed or not. The system's memory. |
| `.stock_screener.env` | SMTP credentials (git-ignored) |
| `data_cache/` | Backtest data cache (git-ignored) |

Useful harness invocations:

```bash
# full validation run
./venv/bin/python backtest_harness.py --interval 1h --subset-size 500 --threshold 11 --regime-filtered

# specific experiments
./venv/bin/python backtest_harness.py ... --sector-analysis   # sector veto, caps, regime exits
./venv/bin/python backtest_harness.py ... --entry-timing      # chase vs retrace-limit entries
./venv/bin/python backtest_harness.py ... --filter-analysis   # expectancy by feature bucket
./venv/bin/python backtest_harness.py ... --recent-only       # same window across intervals
```

---

## 10. Open questions / backlog

- **Caution rule may be too blunt.** Every caution day in late July was triggered
  by the single SPY-below-SMA50 test while breadth stayed healthy (66–72%, far
  above the 40% floor). Six sidelined sessions on one narrow criterion. Worth
  backtesting whether caution should require *two* conditions rather than one.
- **Live edge still unproven.** Needs more sessions before drawing conclusions,
  and before considering any automation of execution.
- **Broker automation (Webull)** — explored 2026-08-03, parked. Webull offers an
  official OpenAPI plus MCP servers. The hosted cloud MCP (read-only, OAuth, no
  API keys) **cannot currently connect from Cursor**: Cursor registers two OAuth
  redirect URIs and Webull's registration endpoint accepts only one, failing with
  `HTTP 417 client.redirectUri.limit.exceeded`. Workarounds if revisited: the
  local MCP server with `WEBULL_TOOLSETS=account,market-data` (needs App Key/Secret,
  stays read-only), or the cloud server from Claude Desktop / ChatGPT.
  For real automation, Webull's **OTOCO bracket orders** are the interesting piece —
  entry, 1×ATR stop, and 3×ATR target submitted as one linked order, with the
  broker enforcing exits so a dead script can't strand a position.
- **No exit automation today.** Stops and targets are managed manually.
