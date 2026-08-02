"""Find Intel-like turnaround analogs: same setup shape, successful recovery, and a
story that is more than a commodity bounce or a takeover.

For each high-recovery ticker, pick the onset closest to the price low (the moment
that most resembles 'I bought when it was cheap'), print the feature snapshot, and
classify the recovery path. Manual narrative tags are applied to the best candidates.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import DATA

pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 120)

# Hand-tagged narratives for the strongest Intel-shaped recoveries.
# Categories:
#   PLATFORM  - franchise temporarily broken inside a structurally growing industry
#   CYCLE     - deep cyclical trough (semis, energy, materials) that always mean-reverts
#   CRISIS    - company-specific scandal/operational failure with industry intact
#   TAKEOVER  - recovery is mostly an acquisition premium (exclude from playbook)
#   MIXED     - elements of more than one
NARRATIVES = {
    'INTC': ('PLATFORM', '2023-2025',
             'Left for dead in semis while NVDA/AI boomed; foundry + USCHIPS + CPU demand thesis.'),
    'AMD': ('PLATFORM', '2015-2016 then 2022',
            'Near-bankrupt vs Intel; Zen architecture turnaround. Later AI GPU lag vs NVDA.'),
    'MU': ('CYCLE', '2008-09, 2015-16, 2019, 2022',
           'Memory cycle troughs; demand always returns. Classic semis mean-reversion.'),
    'WDC': ('CYCLE', '2015-16, 2022-23',
            'HDD/NAND cycle; storage demand tied to cloud/AI data.'),
    'STX': ('CYCLE', '2015-16, 2022-23',
            'Same storage cycle as WDC.'),
    'FSLR': ('PLATFORM', '2011-12, 2017, 2020',
             'Solar OEM battered while renewable policy/demand grew; US manufacturing edge.'),
    'ALB': ('PLATFORM', '2019, 2023-24',
            'Lithium price collapse while EV industry still growing; cost-curve survivor bet.'),
    'FCX': ('CYCLE', '2015-16, 2020',
            'Copper trough; electrification/EV demand thesis underneath.'),
    'NFLX': ('CRISIS', '2022',
             'Subscriber scare + competition; streaming industry intact; ads + password crack.'),
    'META': ('PLATFORM', '2022',
             'Apple ATT wrecked ads; "metaverse waste"; then efficiency + AI ads recovery.'),
    'CMG': ('CRISIS', '2015-16',
            'E. coli crisis; QSR industry fine; food-safety turnaround.'),
    'BA': ('CRISIS', '2019-20, 2024',
           '737 MAX grounding; aerospace duopoly intact; production recovery bet.'),
    'DIS': ('PLATFORM', '2022-23',
            'Streaming losses + Parks COVID hangover; media/Parks franchise still unique.'),
    'PYPL': ('PLATFORM', '2022-24',
             'Fintech growth scare; digital payments industry still expanding.'),
    'NKE': ('PLATFORM', '2023-24',
            'China + wholesale reset; global athletic brand franchise intact — recovery unfinished.'),
    'EL': ('PLATFORM', '2023-24',
           'China travel-retail collapse; prestige beauty industry intact — recovery unfinished.'),
    'BIIB': ('PLATFORM', '2019-21, 2023',
             'Alzheimer drug failures; biotech industry boom; pipeline optionality.'),
    'BBY': ('CRISIS', '2012, 2022',
            'Amazon retail death narrative; CE retail still cash-generative; Huberty turnaround.'),
    'ANET': ('PLATFORM', '2019-20',
             'Cloud capex pause; networking for hyperscalers intact; then AI networking boom.'),
    'TER': ('CYCLE', '2009, 2022-23',
            'Semi test equipment cycle; AI/semicap demand underneath.'),
    'LUV': ('CRISIS', '2022-24',
            'Operational meltdown; airline industry recovered; culture/ops fix thesis.'),
    'C': ('CRISIS', '2009-12',
          'Post-GFC bank rebuild; financial system restored; Fortress balance sheet.'),
    'GNW': ('CRISIS', '2009-12',
            'Mortgage insurance wipeout; housing eventually healed.'),
    'BSX': ('CRISIS', '2008-12',
            'Stent franchise damaged; medtech industry grew; pipeline rebuild.'),
    'PHM': ('CYCLE', '2009-11',
            'Homebuilder trough; housing cycle + demographics.'),
    'OXY': ('CYCLE', '2020',
            'Oil crash; energy cycle + Buffett endorsement.'),
    'DVN': ('CYCLE', '2020',
            'Shale trough; energy cycle.'),
    'NRG': ('CRISIS', '2015-17',
            'Merchant power distress; then retail/generation restructuring.'),
    'EA': ('PLATFORM', '2008-10, 2012',
           'Console transition trough; gaming industry growing; live-services pivot.'),
    'NTAP': ('PLATFORM', '2008-09, 2015',
             'Storage share loss to cloud; enterprise data still growing.'),
    'URI': ('CYCLE', '2015-16',
            'Equipment rental trough with industrial cycle.'),
    'CF': ('CYCLE', '2016, 2019-20',
           'Fertilizer trough; ag cycle.'),
    'NEM': ('CYCLE', '2013-15, 2018',
            'Gold miner trough; gold cycle.'),
    'ALGN': ('PLATFORM', '2019, 2022',
             'Clear-aligner growth scare; dental aesthetics industry intact.'),
    'BBWI': ('CRISIS', '2016-18, 2020',
             'Victoria Secret brand damage; spin + product reset.'),
    'WBD': ('PLATFORM', '2022-24',
            'Streaming merger hangover; content library + industry consolidation.'),
    'WHR': ('CYCLE', '2022-24',
            'Housing/appliance downturn; replacement cycle thesis.'),
    'HRB': ('CRISIS', '2015-17',
            'DIY tax software threat; still cash cow franchise.'),
    'THC': ('CRISIS', '2007-12',
            'Hospital chain distress; healthcare demand intact.'),
    'GEN': ('CRISIS', '2018-20',
            'Symantec/Norton brand reset after Broadcom deal noise.'),
    'EW': ('PLATFORM', '2013?, 2022-24',
           'TAVR growth scare; structural heart industry growing.'),
    'CPRI': ('CRISIS', '2015-16, 2020',
             'Fashion brand fatigue; luxury industry fine — recovery mixed/takeover talk.'),
    'ANF': ('CRISIS', '2009-12',
            'Teen apparel death narrative; brand rebuild.'),
    'PBI': ('CRISIS', '2010-13',
            'Mail/postage decline structural — weak analog (secular decline).'),
    'ECHO': ('TAKEOVER', '2024',
             'EchoStar/Dish special situation — exclude.'),
    'ANDV': ('TAKEOVER', '2018',
             'Andeavor acquired by Marathon — exclude.'),
}


def main():
    with open(DATA / 'clean.pkl', 'rb') as fh:
        d, f = pickle.load(fh)
    close, high, low = d['close'], d['high'], d['low']
    spy = d['etf']['SPY']

    tr = pd.read_csv(DATA / 'analog_onsets.csv', parse_dates=['date'])
    winners = pd.read_csv(DATA / 'analog_winners.csv')

    # For each winner ticker, pick the onset nearest the subsequent local low
    # (most Intel-like entry): among onsets in the year of the best run, take the
    # one with the deepest drawdown / lowest entry price.
    picks = []
    for t, g in tr[tr.ticker.isin(winners.ticker)].groupby('ticker'):
        # onset that produced the best 24m return
        best = g.loc[g.r24.idxmax()]
        # also the cheapest onset within +/- 1y of that best onset
        window = g[(g.date >= best.date - pd.Timedelta(days=365)) &
                   (g.date <= best.date + pd.Timedelta(days=90))]
        entry_row = window.loc[window.entry.idxmin()]
        picks.append(entry_row)

    picks = pd.DataFrame(picks).sort_values('r24', ascending=False)

    # Drop obvious takeovers / artifacts
    picks['narrative'] = picks.ticker.map(lambda t: NARRATIVES.get(t, ('UNKNOWN', '', ''))[0])
    picks['era'] = picks.ticker.map(lambda t: NARRATIVES.get(t, ('', '', ''))[1])
    picks['story'] = picks.ticker.map(lambda t: NARRATIVES.get(t, ('', '', ''))[2])

    print('=' * 160)
    print('INTEL ANALOGS  --  onset nearest the winning low, tagged by narrative type')
    print('=' * 160)
    show = picks[['date', 'ticker', 'narrative', 'entry', 'dd_3y', 'rel_12m', 'ind_ret_12m',
                  'industry', 'r12', 'r24', 'ex24', 'mfe24', 'mae24', 'era']].copy()
    print(show.head(60).round(1).to_string(index=False))

    print()
    print('=' * 160)
    print('BY NARRATIVE TYPE  --  how often each class produces a big recovery')
    print('=' * 160)
    # Tag all winners
    w = winners.copy()
    w['narrative'] = w.ticker.map(lambda t: NARRATIVES.get(t, ('UNKNOWN', '', ''))[0])
    print(w.groupby('narrative').agg(n=('ticker', 'count'),
                                     med_r24=('best_r24', 'median'),
                                     med_ex24=('best_ex24', 'median')).round(1).to_string())

    print()
    print('PLATFORM / CRISIS cases (closest to Intel thesis):')
    plat = picks[picks.narrative.isin(['PLATFORM', 'CRISIS'])].sort_values('ex24', ascending=False)
    for _, r in plat.iterrows():
        print(f"  {r.ticker:<5} {str(r.date.date())}  dd={r.dd_3y:6.1f}%  "
              f"rel12={r.rel_12m:7.1f}  ind12={r.ind_ret_12m:6.1f}  "
              f"-> 24m {r.r24:+6.1f}% (exSPY {r.ex24:+5.1f})  | {r.story[:90]}")

    # Feature contrast: PLATFORM/CRISIS winners vs losers at onset
    losers = pd.read_csv(DATA / 'analog_losers.csv')
    # features of losing onsets
    lose_onsets = tr[tr.ticker.isin(losers.ticker)].copy()
    win_onsets = picks[picks.narrative.isin(['PLATFORM', 'CRISIS', 'CYCLE'])].copy()

    print()
    print('=' * 160)
    print('FEATURE SNAPSHOT AT ONSET  --  PLATFORM/CRISIS/CYCLE winners vs eventual losers')
    print('=' * 160)
    feats = ['dd_3y', 'rel_12m', 'ind_ret_12m', 'ret_12m', 'vol_ann', 'off_52w_low']
    rows = []
    for label, df_ in [('winners', win_onsets), ('losers', lose_onsets)]:
        row = {'group': label, 'n': len(df_)}
        for c in feats:
            row[f'med_{c}'] = df_[c].median()
        rows.append(row)
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    # Industry distribution of winners
    print()
    print('Industry of PLATFORM/CRISIS winning onsets:')
    print(plat.industry.value_counts().to_string())

    # Save a curated shortlist for the review
    curated = picks[picks.narrative.isin(['PLATFORM', 'CRISIS'])].copy()
    curated = curated.sort_values('ex24', ascending=False)
    curated.to_csv(DATA / 'analog_curated.csv', index=False)
    picks.to_csv(DATA / 'analog_picks.csv', index=False)
    print(f'\nSaved {len(curated)} PLATFORM/CRISIS analogs -> analog_curated.csv')


if __name__ == '__main__':
    main()
