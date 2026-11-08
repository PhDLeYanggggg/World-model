# Train the ranker for the fallback it actually uses

The OLS4-trained selector did not improve when moved onto CV, but
its matched hindsight capacity was larger when ranked against CV.
Freeze the same neural proposals. Cross reference (OLS4 or CV) with
training loss (row regression or regret-weighted pairwise ranking).
All four arms see the same causal history, peer summaries and
proposal displacements relative to both baselines. They use the
same initialization, training pairs, and 500 updates. Pair utility
uses the full-grid lower bound and one shared past-derived scale per
query; scaling cannot change the order of candidates within a query.

Use the 108 verified donor forecasters and 27 outer fits from the
nested experiment. Generate each training row's proposal with the
donor that excluded both its source and the outer held source.
Both loss arms receive the same 16 seeded pairs per eligible query
as a sampling pool, then use the same source-balanced draws. Only
training objectives change. Freeze all held predictions before
evaluation. No validation or test threshold search.

Primary: CV-ranked pairwise arm versus leaving CV unchanged,
with exactly one nonidentity-to-OLS4 proposal per eligible
multi-agent query for every switching arm and CV elsewhere. Controls:
all other factorial arms, prior learned OLS4 scores, minimum CV
displacement, observed-prefix replay, random, and CV-ranked
hindsight. Report observed and full-grid lower scores, direct paired
contrasts, hard contribution, known easy harm and undefined support.
Use three seeds and 3,000 source bootstrap draws, averaging seeds
before resampling nine sources. Treat positive lower utility and
improvement over simple controls as requirements for promotion to
a later study, not as a safety or independent-validation pass.

No change to the official 2% rule, the 15/30 split, or data roles.
Use no external, development, or confirmation outcomes. No new
deployment, Stage5C, SMC, metric, or seconds claim.
