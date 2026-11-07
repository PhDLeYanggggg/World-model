# Can Training Supply A Less Harmful Candidate Forecast?

The original CV/EqMotion bank is infeasible in 11 of 102 exposed A recording
views even with a label-aware optimizer. The richer frozen bank is infeasible
in one. Do not tune an action for that held view. Test a prospective training
change for every original source and all three actual seeds (17, 29, 43).

Warm-continue the original 4,000-update fold-2 PCT checkpoint for 300 updates
in each of two arms. PCT is the matched existing temporal comparator and the
cheaper local candidate to adapt. Keep its architecture, absolute forecast
output, initialization, optimizer moments, causal inputs, training groups,
draw stream, batch size 64, fixed LR 1e-4 and gradient clip 5. Do not add a
correction head or generate latent futures. Fitting uses only its original
ten source groups; A target files are opened only after all six checkpoints
and full A forecasts are frozen. Training label access excludes those A
recordings, new DroneCrowd, DUT and other protected cohorts.

Control minimizes the original source-weighted available-step ADE. Treatment
adds `2 * relu(ADE - CV_ADE)` on known easy rows only. Easy in fitting is
positive CV ADE up to its fitting-only 25th percentile, frozen before the
updates. Evaluation uses the previously frozen A slice cuts and the original
2% recording contract; no gain margin or test threshold tuning. Unknown rows
have no ADE supervision and their forecasts still enter the original bounds.

Measure the direct raw treatment-minus-control utility, hard utility and
known easy positive harm; use the full 125,540 held A rows, three seeds, and
same 14 source bootstrap units. For both newly trained paths, also report the
label-aware free-count ceiling after adding the path to the unchanged existing
bank. No saved oracle selections, no passage from a better oracle to a model
claim. Keep the unmodified PCT checkpoint as a secondary forecast comparator.
Checkpoint model/optimizer/sampler/RNG and exact draw counts. The primary
comparison is matched 300-step continuation, not a full from-scratch fit.
No deployment, independent data-role change, metric/seconds, Stage5C or SMC.
