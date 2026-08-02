# Intel analogs — what the winners actually look like

The mechanical screen (damaged + hot industry) produces ~987 situations of which
most lose to the index. This note isolates the subset that *worked* like Intel and
asks what they share that the losers do not.

## Method

1. Find every weekly onset of "Intel shape" in the S&P 500, 2007–2026:
   drawdown ≤ −40% from a 3-year peak ≥ 6 months old, liquid, industry ETF above
   its 200dma with positive 12-month return.
2. Keep tickers whose *best* such onset returned ≥ +80% over 24 months, beat SPY by
   ≥ 40pp, and saw a max favorable excursion ≥ +100%.
3. Tag each by narrative class. Drop takeovers.

Result: **121 price recoveries**, of which **~27 are PLATFORM or CRISIS** — the
classes that match the Intel thesis. The rest are mostly commodity cycles or
one-off mean reversion.

## The two Intel-like classes

| Class | Plain meaning | Intel parallel | Best examples |
|---|---|---|---|
| **PLATFORM** | Franchise temporarily broken *inside* a structurally growing industry | Semis/AI boom, Intel left behind | META 2022, FSLR 2012, ALB 2019, ANET 2019, EA 2012, ALGN 2019, WBD 2024, NTAP 2016 |
| **CRISIS** | Company-specific scandal or operational failure; industry fine | Less exact, but same "market priced death, franchise alive" | CMG 2015–18, BBY 2012, NFLX 2022, BBWI 2019, NRG 2016, LUV 2012 |

A third class, **CYCLE** (MU, FCX, OXY, WDC, PHM), also produces huge recoveries, but
the bet is different: "the commodity/capex cycle turns." Keep it as a secondary
bucket in the screener, labelled honestly, not mixed into the Intel thesis.

## The closest clones

| Ticker | Entry window | Drawdown | Industry backdrop | Thesis in one line | ~24m outcome |
|---|---|---|---|---|---|
| **META** | Oct–Nov 2022 | −76% | Ads/tech damaged short-term, internet intact | ATT + "metaverse waste"; franchise + AI ads recovery | +100pp vs SPY |
| **FSLR** | mid-2012 | −90% | Solar/policy demand growing | Solar OEM left for dead; US manufacturing + demand | +257pp |
| **BBY** | late 2012 | −75% | Retail/CE still cash-generative | "Amazon kills Best Buy"; Huberty turnaround | +211pp |
| **CMG** | 2016–18 | −66% | QSR industry fine | E. coli crisis; food-safety fix | +218pp |
| **NFLX** | May 2022 | −75% | Streaming growing | Subscriber scare; ads + password crack | +202pp* |
| **ALGN** | Aug 2019 | −56% | Dental aesthetics growing | Clear-aligner growth scare | +245pp |
| **ANET** | late 2019 | −44% | Cloud/networking growing | Capex pause; later AI networking | +110pp |
| **EA** | mid-2012 | −55% | Gaming growing | Console trough; live-services pivot | +149pp |
| **ALB** | mid-2019 | −55% | EV/lithium demand growing | Lithium price collapse; cost-curve survivor | +195pp |
| **WBD** | 2023–24 | −77% | Streaming consolidating | Merger hangover; content library | +235pp |
| **INTC** | 2024–25 | −44 to −60% | Semis/AI booming | Foundry + CHIPS + CPU demand | in progress / realized |

\*NFLX's cleanest Intel-shape onset in the data is May 2023 (already off the lows);
the true emotional low was May 2022 at −75%.

## What does *not* separate winners from losers

At the onset date, PLATFORM/CRISIS winners and eventual losers look almost the same:

| Feature (median) | Winners | Losers |
|---|---|---|
| Drawdown from 3y peak | −53.5% | −50.7% |
| Relative 12m vs industry | −40pp | −36pp |
| Industry 12m return | +5.6% | +11.2% |
| Realized vol | 41% | 33% |
| Distance off 52w low | +10.5% | +15.3% |

Insider buying is also weak: only **26%** of these winners had any open-market
Form 4 purchase in the 90 days before onset, and only **11%** had a 3+ insider cluster.

**Price and insider data get you into the right room. They do not pick the chair.**

## What *does* seem to separate them (qualitative checklist)

Every strong PLATFORM/CRISIS analog had most of these, in plain language:

1. **The industry is structurally growing, not just bouncing.** AI, streaming, EVs,
   cloud networking, gaming, solar policy — multi-year demand, not a 6-month ETF pop.
2. **The company still has a franchise or strategic asset.** Brand (CMG, NKE),
   installed base (META, NFLX), manufacturing position (FSLR, INTC), IP/pipeline
   (BIIB, EW), or national-strategic role (INTC).
3. **The damage is temporary or fixable, not secular.** Food safety, ad-targeting
   shock, console transition, lithium price, execution miss — not "the product is
   obsolete and no one needs it" (Pitney Bowes is the cautionary case that still
   "worked" on price but for the wrong reason).
4. **There is a concrete recovery mechanism**, not hope. New architecture (AMD Zen),
   new CEO playbook (BBY Huberty), product/policy catalyst (FSLR, CHIPS Act),
   business-model fix (NFLX ads), cost program (META Year of Efficiency).
5. **Outside recognition of the thesis.** Articles, analyst notes, or social
   consensus that the gap between industry and company is temporary — the same
   confirmation you saw on Intel.
6. **Survivability.** Cash, credit, or strategic importance sufficient to fund the
   turn. Deep damage in a levered no-franchise name is how this screen dies.

## Failed cousins (same shape, wrong outcome)

Useful anti-examples from the loser list and incomplete recoveries:

| Ticker | Why it looked Intel-like | Why it wasn't |
|---|---|---|
| **EL**, **NKE** (2023–24) | Prestige brand / global franchise, China scare | Damage may be structural (China, wholesale), recovery unfinished or slow |
| **PYPL** (2022–24) | Fintech industry growing, stock crushed | Competitive position eroded; "temporary" was partly permanent |
| **DIS** (2022–23) | Parks + IP franchise | Streaming math stayed hard; recovery muted vs peers |
| **PENN**, **CZR** | Gaming/leisure reopening | Leverage + competitive intensity; no durable franchise edge |
| **OGN** | Pharma spin with industry backdrop | Pipeline/balance-sheet reality weaker than narrative |

## Implication for the recommendation algorithm

Do **not** ask the algorithm to emit "buy." Ask it to:

1. **Filter** mechanically into a short weekly candidate list (Intel shape +
   survivability floors).
2. **Score** each candidate on the six qualitative pillars above, using whatever
   inputs we can automate (news headlines, short interest, balance-sheet proxies,
   insider activity as a *soft* bonus) and leaving explicit blanks for your judgment.
3. **Present** a ranked watchlist with the checklist filled in, so you make the call
   the way you did on Intel — with the screen doing the scanning, not the deciding.

That is a recommendation system, not a trading signal. It accepts the research finding
that the mechanical edge is zero, and uses the analogs to define *what to look for*
once the filter has done its job.
