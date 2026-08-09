# How to use the Intel funnel

Operating manual for the mid-term recommendation loop. For *why* it is built this
way see `FUNNEL.md` (design), `PLAYBOOK.md` (what was tested and failed) and
`ANALOGS.md` (what separated the historical winners). This file is only about
running it.

## What this is

A weekly screen that finds badly damaged large caps inside healthy industries,
then assembles the evidence you need to judge whether the damage is fixable. It
was built from the INTC 2025 trade.

It does **not** tell you what to buy. That is not modesty — `validate_stage2.py`
tested whether the score ranks forward returns and found nothing: the information
coefficient is indistinguishable from zero and the score quintiles slope the
wrong way. The ordering in the output arranges the page and nothing more. The
part that has a plausible edge is your judgment on two questions the data cannot
answer, and the loop exists to put good evidence in front of those questions.

Expect roughly 30–50 candidates a week (37 on 2026-08-07, 46 the week before),
of which you should act on very few.

## One-time setup

```bash
cd /Users/juliengaubert/cursor/finance
export SEC_USER_AGENT="Julien Gaubert gaubertjulien13@gmail.com"
```

SEC blocks generic user agents. Without this, fundamentals silently come back
empty. Add it to your shell profile so you stop thinking about it.

## The weekly run

Designed around a Friday close, so Saturday or Sunday fits. Two parts: refresh
the data, then read.

### 1. Refresh (~12 minutes, once a week)

```bash
./venv/bin/python research/fetch_prices.py                # ~3 min
./venv/bin/python research/run_experiments.py --rebuild    # ~8 min
./venv/bin/python research/factor_power.py                 # ~1 min
```

**`--rebuild` is not optional.** Without it, `run_experiments.py` loads the
cached `features.pkl` and your fresh prices never reach the panel the dossier
reads. Everything still runs, prints no error, and quietly reports last week's
candidates at last week's prices. This bit us on 2026-08-08.

### 2. Read (~seconds, after the first run)

```bash
./venv/bin/python research/dossier.py --top 12
```

Writes `research/data/dossier_YYYYMMDD.md` plus a `.json` twin. The first run
ever also builds a company-profile cache across the whole index, which takes
about two minutes; after that it is instant and refreshes itself monthly.

Useful variants:

| Command | Why |
|---|---|
| `--top 20` | Widen the read |
| `--no-news` | Skip headline fetching when you only want the numbers |
| `--refresh-profiles` | Force market caps and analyst targets to refetch |
| `--asof 2024-05-10` | Reconstruct a past week (see the warning below) |

## Reading a dossier

Each name gets one section. In the order they appear:

**What it does** — one sentence. If this is missing, the ticker symbol is stale;
see Troubleshooting.

**Industry** — the reported industry and the three largest competitors with
their 12-month returns. If the competitors are down as much as the candidate,
you are looking at a sector problem and the setup does not apply.

**Setup** — drawdown, divergence, position against moving averages, volatility.

**Why it fell** — a mechanical read of the decline, in three parts. First,
whether the *business* shrank or only the *multiple* did: revenue growing while
the price halves means the market re-rated the name rather than the earnings
breaking, and your job becomes deciding whether that re-rating was warranted.
Second, whether the fall arrived as a few discrete events or a steady grind —
concentrated drops give you specific dates to research, a grind usually means a
slow change of mind. Third, which themes the recent coverage clusters on. It
names dates and magnitudes, and flags which of the worst days were earnings
reactions. It does **not** tell you the actual cause; that still needs the
headlines and your own reading.

**Divergence** — how far behind its return-correlation peer group the name has
fallen. Note this bucket is statistical, not a business grouping: it files ORCL
under XLF. Use it for the size of the gap, not for who the rivals are.

**12-month price trend** — a sparkline of weekly closes with the 52-week
extremes and four quarterly legs. This is the section that distinguishes a name
that bled all year from one that bottomed months ago and is recovering. Watch
for the flag on a low set within the last 90 days: the fall may not be over.

**Financials** — SEC XBRL, point-in-time. **Check `period_end`.** If it is more
than a couple of quarters old, the ticker mapping is broken and every number in
the block is wrong.

**EPS vs industry** — earnings growth and the multiple against real peers. The
combination you want is earnings holding up while the price collapses. If
earnings lag peers too, the discount may be deserved, and your pillar 4 story
has to explain why earnings re-accelerate rather than why the stock looks cheap.

**Reference levels** — three anchors, not a target: analyst consensus,
retracement toward the 3-year peak, and the price implied by peer P/S. They
disagree, often by a lot, and the spread is the point. Each states the
assumption it rests on, so pick the one your thesis actually implies.

**Insiders** — presence is a mild positive, absence means nothing. Only 26% of
historical winners had any.

**Headlines** — raw material for the two pillars below.

## Making the decision

Pillars 1, 2, 3 and 6 arrive pre-filled and are context. Two are left blank
because they are the job:

**Pillar 4 — a concrete recovery mechanism.** Name in one sentence the specific
thing that reverses the damage: a new CEO with a stated plan, a restructuring
already underway, a product cycle, a regulatory or subsidy change, an activist
with a position. "It is too cheap" is not a mechanism. **If you cannot name one,
pass.** This is the pillar the analog study credits above all others.

**Pillar 5 — outside recognition.** Is anyone else making this case? You are not
looking for consensus, which would mean the price already moved, but for
evidence you are not alone in a fantasy.

Red flags worth a pass regardless of how good the numbers look:

- Revenue falling *and* margins compressing — secular decline in a turnaround
  costume, whatever the headlines say
- The `CYCLE` tag — commodity cycles are a different animal from the
  PLATFORM/CRISIS setup this was built for, and deep drawdowns there are usually
  the cycle rather than a fixable company problem
- Peers down just as much — sector problem, not a company problem
- A 52-week low set in the last few weeks with no mechanism named yet

## Sizing and selling

The outcome distribution for this pool is venture-shaped: the median candidate
loses money and a few return +150% or more. The correct response is **many small
positions held long**, not concentration into your favourite. Concentration is
the intuitive move after a win like INTC and it is precisely the wrong lesson to
draw from it — the same distribution that produced that win produces a majority
of losers.

Selling is the genuinely hard part and nothing here automates it. What helps:
decide the exit *when you enter*, and write it into the journal thesis. Two
honest exit rules, both better than watching the price daily:

- **Thesis-based:** sell when the mechanism you named in pillar 4 either
  completes or is disproven. This is the one that matches how the money is
  actually made.
- **Level-based:** pick one of the three reference levels at entry — usually
  halfway back to the prior peak — and take something off there.

Horizon is months, not days. The default journal horizon is 180 days.

## Logging and reviewing

Log every decision, including passes. Passes are what make the review able to
tell whether your judgment is separating names, rather than whether the screen is.

```bash
./venv/bin/python research/journal.py add --ticker BSX --decision buy \
    --thesis "Recall is contained and one-off; margins already recovering" \
    --mechanism "New cost program under new management" \
    --conviction 4 --size 3 --horizon 180

./venv/bin/python research/journal.py list
./venv/bin/python research/journal.py review
```

When you sell, close the position rather than leaving it open. Otherwise `review`
keeps scoring the stock over the full horizon and never sees the trade you
actually made:

```bash
./venv/bin/python research/journal.py close --ticker BSX --price 61.40 \
    --reason "Thesis played out; margin recovery is now consensus"
```

`review` then measures the realised fill over the window your money was at risk,
against SPY over that same window.

`review` scores every decision against SPY over its intended horizon, compares
buys against passes, and once you have five or more buys checks whether your
conviction rating correlates with outcomes.

### Never hand-edit the file

Judgments are appended, facts are corrected. Changing your mind means logging a
second entry — a watch that becomes a buy, a buy you decide against — never
overwriting the first. Each row is worth something only because it was written
before the outcome was known, so editing a past `decision` or `thesis` turns the
review into a measure of hindsight rather than judgment.

Facts that were already true at entry can be fixed, and `edit` exists for that:

```bash
./venv/bin/python research/journal.py edit --ticker BSX --field price --value 49.55
```

Editable fields are `price`, `size_pct`, `horizon_days`, `notes` and the three
exit fields. The command refuses `decision`, `date`, `thesis`, `mechanism`,
`conviction` and `pillars_met` on purpose. Use `--date-of` or `--decision` to
disambiguate when a ticker has several entries.

Editing the CSV in a spreadsheet also reformats the date and price columns,
which is its own source of corruption.

**Nothing in that output means anything until roughly 20–30 resolved decisions.**
With a payoff this skewed, small samples are noise. Treat it as a habit for the
first six months.

The journal lives at `research/data/decision_journal.csv` and is deliberately
**not** version-controlled, because this repository is public and the file
records your positions, sizing and reasoning. It is also the only artifact whose
value comes entirely from accumulating over time. Back it up somewhere private.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Candidates look like last week's | `run_experiments.py` ran without `--rebuild` | Rerun with the flag |
| "Financials not available" or a `period_end` years old | Ticker symbol changed (FISV → FI) so the SEC lookup hits a dead record | Verify the current symbol; treat that block as unusable |
| No "What it does", no competitors | Same stale-symbol cause on the yfinance side | As above |
| Fundamentals empty for everything | `SEC_USER_AGENT` not set | Export it |
| First run seems stuck for minutes | Building the index-wide profile cache | Expected once; instant afterwards |
| Analyst targets look stale | Profiles cached up to a month | `--refresh-profiles` |

**Backdated runs leak the future.** With `--asof`, prices, trends and peer
comparisons respect the date, but descriptions, market caps, analyst targets and
P/S ratios are fetched live and describe today. The dossier prints a warning when
this applies. Never use a backdated Reference Levels section to judge how a name
would have looked at the time.

## Where things live

| Path | What |
|---|---|
| `research/data/dossier_YYYYMMDD.md` | The weekly briefs |
| `research/data/watchlist_YYYYMMDD.md` | Full ranked table from `recommend.py` |
| `research/data/decision_journal.csv` | Your decisions (local only) |
| `research/data/clean.pkl` | The panel everything reads |
| `research/README.md` | Map of every script in the directory |
