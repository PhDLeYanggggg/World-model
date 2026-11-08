# Cross-Query Ranking Under a Fixed Recording Budget

**Outcome: the cross-query loss does not qualify an online controller.**
This is exposed OpenDD development evidence from seven locations, 202
distinct recordings and three training seeds, not independent confirmation.
The nested action menu, observation and prediction horizons (8 and 12
native steps), chronological credits and CV-relative 2% recording rule
are unchanged.

For each training recording, past-only prefix and pair-excluded gain
models propose two causal anchor plans. Candidate pairs move the same
number of edits from a selected query to an unselected query in the
same recording. We reject pairs that cannot replay the original
chronological budget. All learned features and nominations for a donor
exclude both that donor and the outer location. The point-only and
swap models have the same architecture, initialization, point loss,
candidate bank and frozen factorized risk at decision time. Swap training
adds a gain-difference loss and a signed ranking term.

**Partial labels.** If the observed paired difference is `d` and the
sum of distances between the two forecasts at missing steps is `u`,
then the true difference is in `[d-u, d+u]` by the triangle inequality.
For the two disjoint query edits, missing disagreement adds; a shared
unchanged path contributes zero. The direction target is
`sign(d)*max(abs(d)-u, 0)`, up to the common normalization of the pair.
Zero is omitted from ranking rather than treated as a hard preference.
The interval is sharp if missing positions may vary independently in
unconstrained Euclidean space; it need not be sharp with physical path
constraints. This adapts known [decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a.html)
to equal-budget forecast swaps and missing labels. It is not a claim
that ranking feasible solutions is new.

| Policy | Online edits | Online joint pass | Joint pass at 1,457 exact matched edits |
| --- | ---: | ---: | ---: |
| Observed-prefix gain | 1,340 | 56/606 | 68/606 |
| Frozen motion gain only | 1,457 | 61/606 | 60/606 |
| Frozen motion gain and factorized risk | 1,614 | 63/606 | 73/606 |
| Point gain and frozen factorized risk | 1,645 | 62/606 | 60/606 |
| Point plus swap loss and frozen factorized risk | 1,595 | 46/606 | 70/606 |

Joint pass is original risk pass and positive conservative gain versus
both CV and OLS4. The exact-count comparison uses all causal scores
in the recording to match the frozen gain-only counts without labels;
it is a retrospective attribution, not online performance. At exact
counts, swap versus point direct gain is **-0.026 pp [-0.045, -0.009]**
and swap versus frozen factorization is **-0.042 pp [-0.099, -0.007]**.
Online swap versus point is -0.016 pp [-0.084, +0.031]. All intervals
resample seven locations after averaging seeds (3,000 draws).

**The loss did not improve the ranking it targets.** Of 12,641 sampled
held pairs, 11,805 have a certifiable direction (93.39%). Point regression
ranks 54.84% correctly; swap training ranks 53.96%, and frozen motion
ranks 53.37%. These pairs overlap and are not independent. Exact
matched allocation regret against a hindsight conservative-CV-gain
oracle (not risk-qualified) is 0.319 pp for point, 0.326 for swap,
0.310 for frozen factorization and 0.307 for prefix. The oracle fixes
the same edit totals and token rule. Low concordance even in certain
pairs argues against missing labels alone explaining the weak result.

`fresh_run`: 42 arm64 CPU Torch heads, 33,600 updates, three seeds, four
threads and zero workers. `cached_verified`: frozen controls and exact
checkpoint replay with zero updates. Two setup checks stopped before
training (serialized identity type and batch factory); the registered
run is the corrected one. No deployment, Stage5C, SMC or confirmation
change. [Metrics](analysis.json) | [Setup](registration.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_swap_rank.py
.venv-pytorch/bin/python scripts/run_m3w_recording_swap_rank.py --replay
```
