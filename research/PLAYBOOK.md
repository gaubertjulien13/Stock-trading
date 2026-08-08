# The Laggard Playbook — v1.1

Derived from the INTC 2025 trade. Horizon: weeks to months.

**Portfolio trading signal: FAILED** (see validation below). Do not auto-trade Stage 1 alone.

**Recommendation funnel: APPROVED** — see `FUNNEL.md` and `recommend.py`. Stage 1
scans; Stage 2 scores; you decide. Analogs that define the shape: `ANALOGS.md`.

## The trade being generalized

> "I invested in Intel when the price was low last year. The AI industry was booming and
> Nvidia was booming, demand for CPUs and not only GPUs was expected to grow. I expected
> Intel, a solid company, to be able to turn things around. I also saw articles and social
> posts mentioning similar expectations."

Decomposed, this is a bet that **a temporarily broken company inside a structurally healthy
industry will re-rate**. The claimed edge is not "cheap stock" but the *divergence*: the
industry is fine, the company is not, and the market has priced the company as if the
industry were also dying.

## The five pillars, as originally drafted

| # | Pillar | Plain meaning | INTC in mid-2025 |
|---|--------|---------------|------------------|
| 1 | **Damage** | Price far below its own multi-year high | −60% from 3y peak |
| 2 | **Industry tailwind** | The industry is in a real uptrend | SMH +21% y/y, above 200dma |
| 3 | **Divergence** | The stock is the *laggard*, not a sick industry | INTC − SMH = −57pp over 12m |
| 4 | **Survivability** | It can live long enough to turn | Mega-cap, real revenue, strategic asset |
| 5 | **Turn confirmation** | The decline has demonstrably stopped | Reclaimed 200dma Aug 2025 |

## Verdict

Tested on S&P 500 constituents, 2007 → July 2026, using point-in-time index membership
so that companies which later failed are still visible on the dates they were investable.

**987 situations** matched pillars 1–4 (down ≥40% from a 3-year peak at least 6 months old,
liquid, index member, in an industry above its own 200dma with a positive 12-month return).

| Measure | Result |
|---|---|
| Median 12-month return | +6.1% |
| Median 12-month return **vs S&P 500** | **−7.5pp** |
| Share beating the S&P over 12 months | 39% |
| Median worst drawdown in the 12 months after entry | −19.5% |
| Worst drawdown, 1 idea in 10 | −53.1% |
| Traded as a 20-name book, 2008–2026 | **3.2% CAGR vs 11.2% for SPY**, −68% max drawdown |

The industry-agnostic version (drop pillar 2) is worse still: **−1.4% CAGR, −86% drawdown**.
$1 became $0.77 over eighteen years while the index turned it into $7.17.

### Pillar 5 was the hope, and it does not work

The v0.1 draft argued the 200dma reclaim was "the whole ballgame", based on INTC showing a
median 6-month return of +50% across its own 10 reclaim signals. Across all 987 ideas,
holding the candidate list fixed and varying **only** the entry date:

| Entry rule | Acted on | Median wait | Median 12m vs SPY | Beat SPY | Median worst dip |
|---|---|---|---|---|---|
| Buy immediately | 100% | 1 day | −7.5pp | 40.5% | −19.5% |
| Wait for 200dma reclaim | 92% | 19 days | −8.0pp | 39.2% | −19.3% |
| Wait for higher low + 50dma | 88% | 33 days | −6.7pp | 41.5% | −19.4% |

Waiting buys nothing. It does not improve the return and it does not reduce the drawdown.
The INTC result was a single-name artifact.

### Exits do not rescue it either

824 entries taken on the reclaim, varying only the exit:

| Exit | Median | Mean | Median vs SPY | Win rate |
|---|---|---|---|---|
| Hold 6 months | +6.5% | +8.2% | −3.1pp | 60% |
| Hold 12 months | +8.1% | +13.3% | −8.0pp | 58% |
| Hold 12m, hard stop −20% | **−13.0%** | +9.5% | −13.1pp | 45% |
| Hold 12m, trailing stop −20% | −4.5% | +5.7% | −8.3pp | 43% |
| Hold 12m, exit on losing the 200dma | −5.4% | +7.2% | −7.3pp | 36% |

Stops are actively harmful here. These names are volatile enough that a 20% dip is routine,
so the stop fires on noise and books the loss immediately before the recovery.

### The factors carry no information

Rank correlation with forward 6-month excess return, measured inside the setup:

| Factor | Spearman IC | Verdict |
|---|---|---|
| `dd_3y` (how beaten down) | 0.01 | noise |
| `rel_12m` (the laggard factor) | 0.02 | noise |
| `sma200_slope` | 0.02 | noise |
| `off_52w_low_pct` | 0.11 | sign flips by era; unusable |

The two factors that *are* the Intel thesis have no predictive power.

## The momentum result, and why it is also not real

Buying the strongest 12-month performers — the opposite of the Intel logic — appeared to
beat SPY by 2.6 points a year. It does not. That book took 20 names **alphabetically** from
the ~74 that qualified on a typical date, and the alphabet was doing the work.

| Positions | Mean edge vs SPY | Gap between luckiest and unluckiest draw |
|---|---|---|
| 10 | +0.17% | 5.07pp |
| 20 | +0.66% | 3.59pp |
| 50 | −0.08% | 0.91pp |
| 100 | −1.34% | 0.95pp |

Widen the book until arbitrary selection stops mattering and the edge is **−0.08% a year**,
with a Newey-West t-statistic of 0.00. Across 25 random draws of a realistic 10-name book:
mean excess −0.01% a year, standard deviation 1.80%, 13 of 25 beating the index. A coin flip.

## Does it work outside large caps? No — it gets worse

Large caps are the most heavily researched segment, so the null result above says
little about smaller companies, where anomalies are supposed to survive. Point-in-time
membership was rebuilt for the S&P MidCap 400 and SmallCap 600 from their change logs
and 1,149 additional companies downloaded, including delisted ones.

**Usable windows are limited by the change logs, not by choice.** The MidCap 400 log
starts 2012-01, the SmallCap 600 log starts 2019-12. Before those dates the
reconstruction silently degenerates into "today's members", which is the exact
survivorship bias this method exists to avoid. Tests therefore run 2013→2026 for mid
caps and 2020→2026 for small caps.

Benchmark for every row is an **equal-weight index of the same eligible universe** —
what picking at random from the same pool would have given. Note the median forward
return is negative versus that benchmark for *every* setup including the baseline,
because the median stock always trails the average stock. **Compare each row to the
baseline row, not to zero.**

### S&P MidCap 400, 2013–2026 (median 304 eligible names per day)

| Setup | Median 12m vs peers | Portfolio CAGR | CAGR vs peers | Luck spread |
|---|---|---|---|---|
| Baseline: every eligible name | −3.6pp | 13.2% | −0.2% | 8.0pp |
| Short-term reversal (worst 10% over 1m) | −5.8pp | 14.6% | +1.2% | 5.8pp |
| Momentum, top 20% over 12m | −5.2pp | 12.8% | −0.7% | 3.3pp |
| Low volatility + above 200dma | −2.7pp | 11.1% | −2.4% | 3.8pp |
| Near 52-week high | −2.5pp | 10.3% | −3.1% | 3.1pp |
| **Intel playbook** | **−12.3pp** | **9.8%** | **−3.7%** | 3.6pp |

Only 38% of Intel-shaped setups beat their peer group over 12 months. Nothing beats
the baseline by more than its own luck spread.

### S&P SmallCap 600, 2020–2026 (median 440 eligible names per day)

| Setup | Median 12m vs peers | Portfolio CAGR | CAGR vs peers | Luck spread |
|---|---|---|---|---|
| Baseline: every eligible name | −5.8pp | 16.7% | −1.5% | 13.7pp |
| Momentum, top 10% over 12m | −3.9pp | 20.1% | +1.9% | 7.6pp |
| Near 52-week high | −2.9pp | 15.3% | −2.9% | 5.2pp |
| Low volatility + above 200dma | −4.3pp | 11.0% | −7.2% | 5.1pp |
| **Intel playbook** | **−12.0pp** | **15.9%** | **−2.3%** | 8.0pp |

Six years is too short to conclude anything: the luck spread on the *baseline* is 13.7
points a year. Treat this table as suggestive only. What it does not show is any sign
of the playbook working.

**The pattern across all three size segments is the same, and it steepens as companies
get smaller.** That is what you would expect if a beaten-down small company is more
likely to be genuinely dying than temporarily broken. Intel — a mega-cap with a
strategic national-security role and government money behind it — is close to the most
survivable company the screen could ever return, and generalizing from it is exactly
the error.

## Insider buying: real, but with a half-life of days

Form 4 filings are the only signal in this study that is point-in-time by construction.
Every SEC quarterly insider dataset from 2006Q1 to 2026Q1 was downloaded
(`fetch_insiders.py`), filtered to open-market purchases (transaction code P with an
acquisition flag), and **re-valued at the market close on the filing date** — filers
enter prices as absurd as $250,000,000 per share, and 7.4% of records are inconsistent
enough with the market to discard. That leaves 30,716 clean purchase events in the
investable universe. Signals use the filing date, never the transaction date.

### The announcement effect is unambiguous

S&P 500, 2007–2026, 7,254 events, measured against an equal-weight index of the universe:

| Trading days after filing | Mean excess | t-stat |
|---|---|---|
| 1 | +0.34% | **12.8** |
| 2 | +0.39% | **10.9** |
| 3 | +0.35% | **8.3** |
| 5 | +0.36% | **6.7** |
| 10 | +0.27% | **3.8** |
| 21 | +0.09% | 0.9 |
| 42 | −0.17% | −1.1 |
| 63 | −0.26% | −1.5 |

A placebo run of the identical measurement on random days returns +0.04% at 5 days
(t = 0.7), confirming the dates and prices are aligned and the effect is not an artifact.

**The edge is gone within about three weeks.** Past a month there is no signal. (The
long-horizon numbers drift slightly negative for a mechanical reason — a daily-rebalanced
equal-weight benchmark beats the average buy-and-hold stock — and the placebo shows the
same drift, so read those rows as "nothing", not "negative".)

### Held for months, it does nothing

Annualised return, 20 equal-weight names, 6-month holds, mean of five random draws:

| Rule | Large caps | Mid caps | Small caps |
|---|---|---|---|
| Equal-weight universe (peers) | 12.1% | 13.5% | 18.2% |
| Cluster: 3+ insiders buying | 8.8% | 10.0% | 13.4% |
| Cluster: 5+ insiders buying | 9.4% | 9.0% | 17.4% |
| **No insider buying at all** | 11.2% | 13.8% | 13.6% |
| Insider cluster + the Intel shape | −4.1% | 5.6% | 9.1% |

Buying where insiders clustered underperformed buying at random in five of six
combinations, and names with *no* insider activity did better than names with it in two
of three. Adding insider confirmation to the Intel setup did not rescue it.

**This is the one genuine edge found anywhere in the study, and it is incompatible with
the horizon originally wanted.** Capturing 0.35% requires same-day or next-day execution
and many small trades; 20bps of round-trip friction eats more than half of it.

## Why the original story was so convincing

1. **The sample was one.** The rule was read off a trade that worked, then tested on that
   same trade. That cannot fail.
2. **The dead are invisible.** 180 companies in the index since 2007 have no recoverable
   price history — Lehman, SVB, First Republic, Fannie Mae, JCPenney, Kodak. Every one was
   a beaten-down name in an industry that looked survivable. Their absence biases every
   number above *upward*.
3. **The regime flattered it.** 2023–2026 was an enormous tech bull market, and INTC's
   recovery sits inside it.
4. **The measurable part was never the thesis.** Drawdown, industry trend and relative
   weakness are what a computer can see, and they are noise. Whether CPU demand would
   follow GPU demand, and whether that management could execute, are judgements.

## What was tested

- **Universe:** every S&P 500 member 2007→2026, plus MidCap 400 2013→2026 and SmallCap
  600 2020→2026, membership reconstructed month by month from the index change logs so
  that deleted names remain visible on dates they were investable
  (`build_universe.py`, `build_universe_smid.py`). 2,306 tickers in total.

  | Index | Ever a member | Left the index | Price history recovered | Blind spot |
  |---|---|---|---|---|
  | S&P 500 | 841 | 338 | 158 (47%) | 180 |
  | S&P 400 | 940 | 540 | 360 (67%) | 180 |
  | S&P 600 | 1,077 | 474 | 272 (57%) | 202 |
- **Prices:** daily split/dividend-adjusted open, high, low, close, volume (`fetch_prices.py`).
  Series with implausible single-day moves >150% are truncated at the artifact.
- **Industry:** assigned by trailing 250-day return correlation against 15 sector ETFs,
  recomputed every 21 days — never from today's sector labels, which would leak the future.
- **Benchmarks:** SPY *and* equal-weight RSP. An equal-weight book must be compared against
  equal-weight, or it takes credit for a size tilt.
- **Execution:** signal on the close, fill at the next session's open, 10–20bps costs.
- **Split:** parameters chosen on 2008–2017; 2018–2026 touched once at the end.

## What was not, and cannot be, tested here

- **Sentiment.** The "articles and social posts" pillar. No usable free historical archive
  exists. Forward-testable only.
- **Fundamentals.** yfinance returns *today's* balance sheet, not what was knowable then,
  so any quality filter built on it leaks the future. This is no longer a hard limit:
  `fetch_fundamentals.py` pulls point-in-time figures from SEC XBRL company facts, dated
  by filing. It was written for the dossier layer and has not been used to build or test
  a quality filter, so the conclusions above still exclude fundamentals entirely.

## Reproduce

```
venv/bin/python3 research/validate_assumptions.py  # benchmark, selection rule, delisting assumptions
venv/bin/python3 research/validate_edge.py         # concentration vs luck; is the edge persistent
venv/bin/python3 research/validate_timing.py       # entry timing and exit policy, candidates held fixed
venv/bin/python3 research/smid_test.py             # the same setups in mid and small caps
```

Rebuild the mid/small-cap data first with `build_universe_smid.py` then
`fetch_prices_smid.py` (about 80 seconds of downloads).

## What is left untested

Price data has now been exhausted across all three size segments. Everything that remains
requires data this workspace does not have:

- **Point-in-time fundamentals** — valuation, margins, balance-sheet strength as they were
  *reported at the time*. yfinance gives today's figures only, which leaks the future.
- **Earnings dates and estimate revisions** — post-earnings announcement drift and
  revision momentum are the best-documented anomalies on the weeks-to-months horizon.
  Both need a paid feed with restated history.
- **Sentiment** — the "articles and social posts" pillar. No usable free historical
  archive. Forward-testable only.

Insider transactions have now been tested and are covered above.

## Open decision

No rule tested here earns the right to allocate capital on its own signal at a
weeks-to-months horizon, across three universes, four entry policies, eight exit
policies, ten setup classes and every insider purchase filed since 2006. The honest
directions — a discipline/journal tool around self-generated ideas, pushing the insider
signal to its actual days-long horizon, paying for point-in-time fundamentals, or index
core plus a small conviction sleeve — are laid out in the review canvas.

```
venv/bin/python3 research/fetch_insiders.py   # SEC Form 4 datasets, 2006Q1-2026Q1
venv/bin/python3 research/insider_test.py     # cluster buying across all three universes
venv/bin/python3 research/insider_sanity.py   # announcement effect + placebo control
```
