# Direct causal trajectory forecasting before selector training

The v147 search found that adding fixed paths did not create more
qualified plans at matched counts. I test a new trainable path generator
directly on the full roster, rather than training another action ranker.
It forecasts velocity in a motion-aligned frame: three bounded control
points define a quadratic velocity profile whose integral gives the
twelve requested positions. The untrained model reproduces OLS4.
An ego history encoder sees seven backward differences; a second encoder
sees up to four past-only neighbors in the same query. No scene ID,
future endpoint, centered difference, or future mask enters inference.

The three matched arms are ego-only observed-step loss, ego-plus-peers
observed-step loss, and ego-plus-peers loss with per-missing-step paired
disagreement against OLS4. They share architecture, initial seed, fitting
draws and 1,600 updates per source-role/seed cell. The fixed linear
references are CV, OLS4 and OLS8; two frozen, source-exclusive neural
comparators are fixed-head EqMotion and Past Context Transformer. Their
results are local-protocol replays, not published benchmark reproductions.

The existing three source-role rotations and three seeds cover the same
nine exposed source groups and 24 recordings. Select one learned arm
using only the calibration roles' source-balanced observed gain over
OLS4. Freeze the arm and all held paths before held scoring. Report all
predefined arms, observed and missing-step-conservative accuracy,
available-label ADE/FDE, original CV-relative risk, and paired 3,000-draw
source-group intervals. A direct forecast is a strong candidate only if
both held OLS4 interval lower endpoints are positive. This is not a
deployment test; no threshold or 2% rule is changed. No reserved
DroneCrowd calibration or independent confirmation is used.
