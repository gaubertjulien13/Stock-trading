# research/ — the Intel-style mid-term funnel

A recommendation and evidence tool, **not** a trading signal. It puts damaged
large caps on your desk with the facts assembled; every buy and sell is your call.

Read `FUNNEL.md` for the design, `PLAYBOOK.md` for why the mechanical version
doesn't work on its own, and `ANALOGS.md` for what separated the historical winners.

## Start here

Most files in this directory are finished investigations. Only these are the loop:

| Script | What it does |
|---|---|
| `recommend.py` | Stage-1 filter + Stage-2 score, writes the weekly watchlist |
| `dossier.py` | Evidence brief per candidate: what the company does, industry and competitors, 12-month price trend, fundamentals, reference price levels, insiders, news, pillar checklist |
| `journal.py` | Records decisions and theses before outcomes are known |

```
venv/bin/python3 research/dossier.py --top 12
venv/bin/python3 research/journal.py add --ticker XYZ --decision buy --thesis "..."
venv/bin/python3 research/journal.py review
```

Refresh the data first, weekly — see the "Reproduce / run" section of `FUNNEL.md`.
`run_experiments.py` needs `--rebuild` or the fresh prices never reach `clean.pkl`.

## The one thing to know before trusting the output

`validate_stage2.py` tested whether the Stage-2 score ranks forward returns. It
does not: the information coefficient is indistinguishable from zero and the
score quintiles slope the wrong way. The ranking in the watchlist orders the
page, nothing more. The dossier's value is the assembled evidence, and the edge —
if there is one — lives in pillars 4 and 5, which are left for you to fill in.

## Data pipeline

| Script | Output |
|---|---|
| `build_universe.py` | Point-in-time S&P 500 membership from Wikipedia's change log |
| `fetch_prices.py` | Daily OHLCV for the full universe, including delisted names |
| `run_experiments.py --rebuild` | `features.pkl` — the factor panel |
| `factor_power.py` | `clean.pkl` — the cleaned panel everything else reads |
| `fetch_fundamentals.py` | Point-in-time SEC XBRL financials (used by `dossier.py`) |
| `fetch_insiders.py` | SEC Form 4 open-market buys |
| *(inside `dossier.py`)* | `company_profiles.json` — yfinance descriptions, industries, market caps and analyst targets for the index; built once, entries refreshed monthly |
| `engine.py` | Shared feature/backtest library, imported by most of the above |

Scripts import each other by bare module name and rely on Python putting this
directory on the path, so they run from the repo root but must not be moved into
subfolders without adding explicit path handling.

## Validation record

These back the claims in `PLAYBOOK.md` and `FUNNEL.md`. Keep them: the documents
assert that the mechanical filter has no portfolio edge, and these are the only
way to reproduce that.

| Script | Question it answers |
|---|---|
| `validate_assumptions.py` | Benchmark choice, selection rule, delisting assumptions |
| `validate_edge.py` | Concentration vs luck; is any edge persistent |
| `validate_timing.py` | Entry timing and exit policy, candidates held fixed |
| `validate_stage2.py` | Does the Stage-2 score rank forward returns (no) |
| `portfolio_test.py` | Does a diversified basket beat buy-and-hold |
| `horse_race.py` | Candidate setups compared at the weeks-to-months horizon |
| `smid_test.py` | Does anything work outside large caps |

## Finished investigations

Conclusions already written into the Markdown docs. Kept for provenance; nothing
in the loop depends on them.

`analogues.py`, `find_analogs.py`, `intc_case_study.py` — the Intel trade and its
historical analogs. `recovery_check.py`, `recovery_screen.py`, `tune_momentum.py` —
attempts to turn the surviving discriminator into a rule. `insider_test.py`,
`insider_sanity.py` — whether insider buying predicts returns. `diagnose.py`,
`audit_coverage.py` — data quality and the survivorship blind spot.
`build_universe_smid.py`, `fetch_prices_smid.py` — mid/small-cap universe for
`smid_test.py`. `export_canvas.py`, `export_for_review.py` — JSON payloads for a
one-time strategy review.
