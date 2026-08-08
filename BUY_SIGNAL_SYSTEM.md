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

## 5b. The benchmark result — does this beat just owning SPY?

Added 2026-08-08 (`--benchmark`). Everything in §5 was validated on **expectancy in R
per trade**, which turns out to be the wrong metric on its own: R ignores how much
capital is tied up, for how long, and how many signals a 1–5 position account can
actually take. A strategy can post positive expectancy and still lose to buy-and-hold.

The portfolio test caps concurrent positions, skips signals when full, compounds
realized equity, and compares against two controls — SPY buy-and-hold (what you'd
otherwise do with the money) and random tickers at the same entry times with the same
exits (what picking by coin flip would give).

**At 5 concurrent positions:**

| Window | Split | Strategy | SPY | Edge | Random |
|---|---|---|---|---|---|
| 2024-Q1 | train | +10.25% | +7.07% | +3.18pp | −6.86 ±4.28 |
| 2024-Summer | train | +5.00% | +3.21% | +1.78pp | −9.27 ±4.55 |
| 2024-Q4 | train | +10.09% | +5.97% | +4.12pp | −5.68 ±5.70 |
| 2025-Q1 | train | −4.30% | −5.87% | +1.57pp | −7.39 ±5.64 |
| 2025-Summer | train | −1.20% | +4.44% | −5.64pp | −4.80 ±4.02 |
| **2026-Q1** | **test** | **+27.18%** | +0.41% | **+26.77pp** | +4.47 ±8.63 |

Train edge **+1.00pp**, beat SPY in 4/5 train windows.

**At 3 concurrent positions** (closer to how it's actually traded), the picture inverts:
train edge **−1.63pp**, beat SPY in only **2/5** windows. The entire positive result comes
from the single 2026-Q1 holdout (+34.42pp).

### Conclusions

1. **The edge over SPY is not established.** It flips sign with position count and is
   carried by one window. The random-selection spread is **21–27pp** between the luckiest
   and unluckiest draw; an edge of 1–5pp is not distinguishable from luck at this sample
   size. This is the same standard the `research/` playbook applies, and by that standard
   the answer is no.
2. **The signal is not noise, though.** It beat random selection in **6/6** windows, with
   random consistently at −5% to −11%. There is real short-term timing information; it is
   just not worth more than owning the index once capital lockup is accounted for.
3. **The edge lives entirely in a small tail.** Win 11.8%, stopped 46.1%, timeout 42.1%.
   Average winner +2.95R, average loser −1.05R. Roughly one trade in eight produces
   essentially all the profit.

Point 3 explains the live experience better than anything else. A ~12% hit rate means
**two flat weeks are completely expected even if the edge were real** — the sample is far
too small to judge. It also means discretionary exits are fatal: taking profits early
turns +2.95R winners into +1R while losers stay at −1R, which is enough to erase
everything. The backtested result assumes every trade is held mechanically to its stop,
target, or time stop, which is precisely the part that isn't happening live.

## 5c. Exit selection redone on the portfolio metric

The 1.0×/3.0× exit in §5 was chosen by maximizing R-expectancy. Re-running that selection
under portfolio-return-vs-SPY (`--exit-benchmark`, 41 variants, selection on train windows
only) shows the two metrics rank exits differently, and the adopted exit ranks **9th of 41**
— negative at 5 positions (−0.42pp).

Per-window edge vs SPY, the stability check that matters:

| Exit | Expectancy | @3 positions, by window | @5 positions, by window |
|---|---|---|---|
| **fixed 1.0/3.0** (adopted) | +0.223R | −7.8, +5.5, −1.3, +0.5, −5.1, **+34.4** | +3.2, +1.8, +4.1, +1.6, −5.6, **+26.8** |
| **trail 2.0×ATR** | +0.162R | +1.0, +8.2, +10.1, +3.9, −6.6, **+17.6** | +0.9, +2.3, +5.9, +1.5, −5.2, **+11.8** |
| fixed 0.75/1.5 | +0.146R | −2.9, −2.8, −0.8, +5.0, +4.9, **+10.5** | −4.6, −6.3, +9.7, +1.8, −5.0, **+9.6** |

**The trailing stop has lower R-expectancy but better and far more stable portfolio
results** — positive in 10 of 12 window/position cells, and the same sign at both position
counts in every window. The adopted fixed exit flips sign between 3 and 5 positions in two
windows and is positive in only 3 of 6 at three positions. That is the thesis confirmed
concretely: optimizing R picked an exit that is unstable in a real portfolio.

Mechanically the trail stops out less (36.1% vs 45.7%) and times out more (54.9% vs 42.4%).
It lets winners run instead of waiting for a 3×ATR target that arrives only ~12% of the
time, and it converts "when do I sell?" from a judgment call into a rule.

**Caveats that keep this from being a green light.** Every magnitude here (+1 to +10pp) sits
below the 21–27pp random-selection spread, so none of it is distinguishable from luck. 41
variants were swept, so the top train performer is partly selection noise — the case for the
trail rests on consistency, not on being the highest scorer (it ranked 6th on train). And
2025-Summer is negative for every variant tested, which suggests a regime the system handles
badly rather than a fixable exit problem.

**Recommendation:** if the system is traded at all, a 2.0×ATR trailing stop is better
supported than the current fixed 1.0×/3.0×, and it removes the discretionary sell decision.
It is not evidence that the system beats owning the index.

```bash
./venv/bin/python backtest_harness.py --interval 1h --threshold 11 --subset-size 300 \
  --non-overlapping --regime-filtered --exit-benchmark --max-positions 5
```

```bash
./venv/bin/python backtest_harness.py --interval 1h --threshold 11 --subset-size 300 \
  --non-overlapping --regime-filtered --benchmark --max-positions 5 --benchmark-runs 25
```

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

**Week of Aug 3–7, 2026** (SPY **+3.51%**): 422 signals, 310 non-vetoed. Simulated on real
15m data, non-vetoed alerts averaged **+0.12R** with 58% winners and only 13% stopped —
but **zero of 295 reached the 3×ATR target**, and holding them to Friday returned
**+0.50% against SPY's +3.51%**. Positive in R, badly behind the index in money. The vetoes
again did their job (non-vetoed +0.12R vs vetoed −0.13R).

That is two consecutive weeks trailing SPY, which §5b says is expected noise for a 12%
hit-rate system rather than proof of failure — but it is also not evidence of an edge.

**Intel funnel, same week.** Worth recording because the conclusion is counterintuitive:
the three STRONG names averaged **+1.86%, trailing SPY by 1.65pp**, while the 40 WATCH
names averaged **+5.62%**, beating it by 2.11pp. The ranking was inverted. BSX +5.52% was a
genuine win, INTU +2.90% actually lagged the index despite feeling like one, and FISV
−2.84% was the STRONG name that got skipped. Five days cannot validate a weeks-to-months
thesis, but the funnel's own 19-year backtest already reported no edge (median 12m −7.5pp
vs SPY, 20-name portfolio CAGR 3.2% vs 11.2%).

The Intel funnel's Stage-2 score was subsequently validated over 19 years and found to
carry **no rank information** (IC ≈ 0 at every horizon, quintiles sloping the wrong way,
STRONG's mean driven by three outliers while its median pick loses money). See the
Stage-2 section of `research/FUNNEL.md`. That closes the last untested selection idea in
this repo — every stock-selection approach here has now been measured and none beats the
index.

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

# the one that matters: portfolio return vs SPY and vs random selection
./venv/bin/python backtest_harness.py ... --benchmark --max-positions 5 --benchmark-runs 25

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
