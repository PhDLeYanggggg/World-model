# Native Moving-Drone Working Manuscript

This version separates the native MovingDroneCrowd++ observation8/prediction12
task at stride1 from the earlier European stride12 study and historical SDD
raw-frame studies. Their absolute errors and gains are not pooled.

The [joint path audit](../moving_joint_path_audit_v1/readout.md) adds
paired missing-step bounds, final-step errors and tail measures to the
frozen matched-count comparison. It retains the negative comparison
with the matched additive control and does not change the main PDF.
The standalone [supplement page](joint_path_appendix.tex) was compiled
and visually checked; [build receipt](joint_path_appendix_verification.json).

The [learned sparse exchange ablation](../moving_sparse_quarter_exchange_v1/readout.md)
tests the capacity result with past-only inputs. Donor calibration leaves
the original 15 passing views unchanged despite 19 replacements. The
support term is not a demonstrated method improvement. This supplement
record does not change the compiled main PDF.

The [sparse quarter-amplitude capacity study](../moving_quarter_swap_capacity_v1/readout.md)
separates candidate availability from learned ranking. A single same-query
exchange with labels can repair 26 failing views at positive paired lower
gain, but this does not demonstrate learned recovery. Larger-cap timeouts
remain visible. This study is supplemental evidence, not a new main PDF.

The [native inference rehearsal](../dronecrowd_native_inference_rehearsal_v1/README.md)
documents the common prefix, comparator roster and exact three-seed
kinematic replay. It is an engineering receipt; no external result or
new accuracy claim is included in the compiled paper.

The [matched-count allocation ablation](../moving_quarter_allocation_v1/readout.md)
is a development result for the supplement. With identical candidate costs,
per-query counts and quarter amplitude, adding a harm penalty cuts total-risk
violations from 64 to 56, but does not increase the pass count. Positive
observed-label comparisons against the original roster are not robust to
missing outcomes. The new study is not included in the compiled main PDF.

The [insertion appendix](insertion_appendix.md) collects five
matched development comparisons and the verified action-set
capacity checks. It is generated from score/replay receipts by
`scripts/build_m3w_insertion_appendix.py`. The one-page local PDF
is not committed. See [build verification](insertion_appendix_verification.json).

The abstract and Section 6.6 distinguish hindsight repair capacity,
shortlist coverage, and the behavior of a calibrated past-only policy.
The 102 exposed views are 34 recordings repeated across three forecaster
seeds, not 102 independent recordings. Four frozen analysis files are
bound to both their scoring and replay receipts by the builder.
[Build verification](repair_revision_build.json) records the local PDF
check. No model is refitted, no reserved outcome is opened, and no
threshold or operating policy changes in this revision.

The draft integrates completed support, prefix-loss and past-image cost-head
experiments, including their negative registered contrasts and failed original
recording-risk checks. All twelve past-image actors complete4000updates and
both fixed-design full refits reproduce exactly. Their primary whole12known
effect is-1.804pp and completion lower-10.931pp[-17.826,-5.288];the component
screen fails. These outcomes and frozen-checkpoint diagnostics are included,
with their source hashes. No new outcome scoring is done by the manuscript builder.

The completed query-view experiment adds six full4,000update actors,a full
identity control and one complete first-design treatment refit. All twelve
actor/camera-CV forecast banks and scalars replay exactly. Its primary known
effect is-5.133pp,paired lower-21.447pp[-33.682,-9.911],with3/14positive
known sources. The camera-CV secondary control does not replace this failed
primary. All six final matched training minibatches are worse;this is not a
training-average loss estimate or identification of a unique causal failure.
The draft preserves the failure and original risk/support/NA rules.

This is a working evidence draft, **not submission-ready**. Exact replay and a
compiled PDF do not establish the method contribution or independent evidence.
The original2%/support/NA requirements remain; identity certification is not
introduced. Stage5C execution and SMC remain off.

The bounded-substitution revision includes the matched experiment.
The expanded bank improves utility but passes no original risk view;
uniform quarter displacement passes 29 of 102 at lower utility.
The PDF remains eight pages. Pages 1 to 6 are unchanged and pages 7 and 8
were inspected. Forty scoped tests pass. The supplement text and PDF are
unchanged. [Bounded substitution results](../moving_record_joint_actions_v1/causal_ledger/bounded_bank/readout.md).

The incremental revision adds the negative comparison at identical amplitudes
and the small, posthoc amplitude-reassignment ceiling. The eight-page PDF
changes only page 7, which was visually checked; page 8 and the supplement
are unchanged. Forty-three scoped tests pass.
[Incremental costs](../moving_record_joint_actions_v1/causal_ledger/bounded_bank/incremental/readout.md).

The latest revision reports the wider support diagnostic and the +0.200 pp
matched gain from veto refitting alongside unchanged risk passes and
worsened easy net loss. The eight-page PDF was rebuilt; pages 1 through 6
are unchanged and pages 7 and 8 were inspected. The supplement is unchanged.
[Veto refit](../moving_record_joint_actions_v1/causal_ledger/veto_refit/readout.md).

The separate [utility and harm appendix](utility_tradeoff_appendix.md)
reports the calibration-size, source-stable risk, and directional gain
comparisons at the same query-matched switch count. Each row uses its
registered control. The one-page local PDF was compiled and visually
checked; the main PDF and the mathematical supplement are unchanged.
Build its Markdown and TeX from the replay receipts with
`.venv-pytorch/bin/python scripts/build_m3w_utility_tradeoff_appendix.py`.

The [capacity diagnostic](../moving_record_joint_actions_v1/causal_ledger/prefix_capacity/readout.md)
also has a separate, visually checked vector figure under the local
`utility_tradeoff_build` directory. It contrasts four nested constraint
sets and keeps unresolved solver results visible. It does not replace
the failed learned-policy comparison or change the main manuscript.

The [matched-total timing diagnostic](../moving_record_joint_actions_v1/causal_ledger/timing_capacity/readout.md)
separates timing from intervention volume and contrasts it with two
past-only waiting rules. The local appendix directory contains the
checked `timing_capacity.pdf` figure. The main PDF is unchanged.

The [opening-action ablation](../moving_record_joint_actions_v1/causal_ledger/easy_joint_anchor/readout.md)
compares a joint easy-and-low-harm event with marginal prediction,
and distinguishes missing first-query support from failures to rank
available candidates. Neither calibrated rule improves full observed
risk passes. This is supporting negative evidence, not a main result.

The [stop-versus-wait test](../moving_record_joint_actions_v1/causal_ledger/opening_stop_wait/readout.md)
holds candidate ranking fixed and fits a source-held next-query
continuation head. No calibrated held-source decisions change.
The shortlist recall check separates timing from ranking: positive
supported first-two-query opportunities exist in 15/18 zero-direct
views, but the tested top-ranked choices recover none.

The [useful-action ranker](../moving_record_joint_actions_v1/causal_ledger/opening_pair_rank/readout.md)
compares a pointwise and a pairwise objective on identical draws. The
local appendix figure `opening_rank_recall.pdf` also shows an
eight-slot diversity control. Shortlist gains do not turn into
calibrated policy gains; the main PDF is unchanged.

The one-page [fixed-shortlist appendix](shortlist_appendix.md) brings
bank availability, shortlist recall and final choice into one table.
Its local PDF compiles and was visually checked. Rebuild its source
with `.venv-pytorch/bin/python scripts/build_m3w_shortlist_appendix.py`.
This adds a supporting negative result; the main PDF is unchanged.

Two further fixed-shortlist ablations keep this conclusion intact.
[Past forecast errors](../moving_record_joint_actions_v1/causal_ledger/shortlist_history/readout.md)
show a small matched effect for the short observed-window backtest
but no additional calibrated pass.
[Observed neighbor motion and rollout proximity](../moving_record_joint_actions_v1/causal_ledger/shortlist_interaction/readout.md)
do not improve either calibrated pass counts or early useful-choice
recall, and increase signed easy violations at matched switch count.
These linked tables are development evidence; neither is a positive
main-method result or an independent evaluation. The PDF is unchanged.

The [shortlist follow-up appendix](shortlist_followup.md) compares those
two ablations with the new fixed-budget cost portfolio. The portfolio
finds two different useful openings but loses five retained by the
family-diverse control; neither calibrated rule improves full passes.
Build the bound Markdown and TeX from the exact score replays with
`.venv-pytorch/bin/python scripts/build_m3w_shortlist_followup.py`.
The one-page local PDF under `shortlist_followup_build` was compiled
and visually inspected. The original appendix and main paper are
unchanged.

The [retrieval loss audit](../moving_record_joint_actions_v1/causal_ledger/shortlist_exclusions/readout.md)
separates the cost ranking from agent preference and includes a
predeclared six-plus-two replay with the original head. Five dropped
actions came from families absent at the corresponding queries.
The hybrid restores three losses without adding any calibrated
intervention. This is additional development evidence; the compiled
appendices and main paper are unchanged.

The [within-family supported-gain experiment](../moving_record_joint_actions_v1/causal_ledger/family_supported_gain/readout.md)
recovers six locally useful early winners across the full pre-direct
cohort, but adds no complete pass and increases signed easy violations.
Even retrospectively inserting those winners into their respective
direct records yields zero complete passes. This exposes a gap
between local action labels and the full-record decision objective;
it is not an independent result or a change to the risk contract.

The [recording pruning experiment](../moving_record_joint_actions_v1/causal_ledger/record_pruning/readout.md)
tests deletion alone. Checked hindsight non-loss subsets exist in
30/102 views, including the two existing passes. The causal pruner
has no donor threshold that gains passes while preserving easy risk
and paired lower utility. Its zero changes are not a matched-coverage
efficacy result. Main paper and PDF are unchanged.

[Removal-set supervision](../moving_record_joint_actions_v1/causal_ledger/pruning_teacher/readout.md)
learns from checked fitting-source subsets. Unlike the moment
pruner, it deletes some actions after donor calibration but adds
no held pass. The gain guard loses one existing pass by deleting
selected easy support. This paired transition is evidence against
equating donor calibration or fewer signed easy violations with improved
recording risk; the main paper and PDF are unchanged.

The [causal routing experiment](../moving_record_joint_actions_v1/causal_ledger/pruning_router/readout.md)
estimates repairability before a deletion. First-query discrimination
is weak and neither routed arm adds a pass. A fixed-threshold replay
shows that the decline from 44 deletions to one depends on
recalibration, not the support veto alone. Main PDF is unchanged.

The [support acquisition capacity study](../moving_record_joint_actions_v1/causal_ledger/support_acquisition/readout.md)
compares one insertion, ordered joint edits, free order and the
old shortlist in the 35 views with undefined easy support.
The ordered full bank reaches 33 checked feasible views, versus
18 using the shortlist. A new source-bound vector figure,
`support_acquisition.pdf` in the local `utility_tradeoff_build`
directory, was generated and visually inspected. It is an
appendix diagnostic, not an independent policy result. The main
PDF is unchanged.

The [equal-budget support retrieval experiment](../moving_record_joint_actions_v1/causal_ledger/support_joint/readout.md)
raises checked shortlist capacity from 18 to 20 of the 35 views.
In the donor grid, proposed support acquisitions supply no
observed easy reference in the missing-support stratum; no
operating arm calibrates. The small retrieval benefit is not
evidence of a learned risk improvement. Main PDF unchanged.

The [matched-agent easy-reference experiment](../moving_record_joint_actions_v1/causal_ledger/easy_reference_rank/readout.md)
separates observed-support prediction from intervention. Boosting
improves pooled discrimination, but the top-one effect versus
the stronger joint control is inconclusive and first-query hits
do not improve. It is not admitted to a new policy. The main
PDF and risk contract are unchanged.

The [early-weighted listwise study](../moving_record_joint_actions_v1/causal_ledger/early_reference_listwise/readout.md)
holds the network and fitting draws fixed while changing the
training weight on the first two eligible queries. It worsens
first-query accuracy by 11.35 pp against the uniform objective,
with a negative paired interval. This negative result constrains
the early-support mechanism; no change to the main PDF or policy.

[中文复现说明](reproduction_shortlist_zh.md) covers the latest local
replay, checkpoints, input requirements, and when to use CREATE.

## Files And Reproduction

- `manuscript.template.md`: editable English narrative and formulas.
- `manuscript.md` and `manuscript.tex`: generated from public scalar receipts.
- `source_bindings.json`: pinned evidence and generated-source hashes.
- `build_verification.json`: current PDF and visual-check record.

```sh
.venv-pytorch/bin/python scripts/build_m3w_native_paper_v10.py
python3 /Users/yangyue/.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py \
  data/stage_cvpr2027_experiments/evidence_manuscript_v10/paper_build/main.tex \
  --compiler tectonic --json
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_manuscript.py
```

The local generated PDF and scalar-only figures are under
`data/stage_cvpr2027_experiments/evidence_manuscript_v10/paper_build/`.
No dataset images, third-party style file, raw data, weights or caches are
committed. A six-file portable aggregate package now replays the three completed
native actor primary analyses in a clean stdlib-only process. It preserves the
original bootstrap indices and negative outcomes; it is not complete training
or prediction reproduction, independent confirmation or guaranteed anonymity.
Full model-training reconstruction is still outstanding.

The eight-page PDF records the stopped one-sided information-loss diagnosis,
the literature recheck and the completed replacement dual-view experiment.
Both registered dual-view screens fail despite exact twelve-bank/scalar replay.
The whole-12 known contrasts are +0.828pp versus the matched native branch and
-0.321pp versus the original actor; the completion lower intervals are entirely
negative. Pages 1 through 5 are identical to the inspected previous render;
changed pages 6 through 8 and the new source/training figure were inspected.
Forty-three focused checks pass; the full historical suite was not rerun.
The frozen v9 sources and PDF are unchanged.

The 7 November revision retains eight pages and includes the negative
matched-count pair comparison plus a separate three-seed predictor control.
The full 24-group predictor primary remains not estimable. The 23 supported
groups are a secondary result, not evidence of an independently confirmed
three-seed intervention policy. The named LLM Usage Statement now precedes
References. Pages 6 through 8 were inspected; pages 1 through 5 were identical
to the last inspected render. Thirty-five focused checks passed for this update.

The later seed-disagreement revision adds the source-excluded matched-count
cost experiment. The conservative utility contrast is -3.535 pp with an
entirely negative exploratory source interval. The PDF still has eight pages;
the main text ends on page seven, followed by the disclosure and references.
Changed pages were inspected and 25 focused checks pass. This is still a
working draft, not a deployment or submission claim.

The recording-risk revision adds a direct comparison with OLS4 and the
unchanged selected-reference gate on all 34 exposed recordings and three
forecast seeds. The learned anchor passes no recording view. The eight-page
PDF is checked again: pages 1 through 6 are pixel-identical to the prior render,
and pages 7 and 8 were inspected. The mandatory disclosure still precedes
References. Twenty-seven focused tests pass; no new independent result.

The candidate-ceiling revision records the exact label-aware binary
free/fixed-count comparison and nested-bank limit, not an inference result.
The draft remains eight pages. Pages 1 through 6 match the inspected prior
render exactly; changed pages 7 and 8 were inspected. The supplement TeX is
unchanged and its source binding is refreshed. Twenty-nine focused tests pass.

The three-seed continuation revision records the new easy-objective training
ablation and its conservative utility loss. Both candidate banks still miss
one required view. Eight pages, pages 1 through 6 pixel-identical to the last
checked version and changed pages 7 and 8 inspected. Thirty focused tests pass.

The missingness-aligned continuation adds three matched training runs. It
reduces disagreement with causal CV on unknown rows, but the completion lower
bound versus CV is -40.03 pp and the raw model passes 0 of 102 exposed views.
The rules are unchanged and the model is not promoted. The revised draft
compiles to eight pages; pages 7 and 8 were inspected, and 31 focused tests
pass. The supplement TeX is byte-identical; only its source binding changed.

The direct kinematic revision reports matched three-seed history-only and
peer-conditioned forecasts. Either recovers the last view for a label-aware
oracle, while the raw forecasts still fail every view. This is not an
independent allocation result. The revised PDF remains eight pages; pages 1
through 6 are identical to the previous render, and pages 7 and 8 were
inspected. Thirty-four focused tests pass. Supplement TeX is unchanged.

The matched multi-action revision adds the joint-versus-marginal negative
contrast and the filter-first gain with failed original risk. No independent
result or rule change. Eight pages, pages 1 through 6 identical to the last
render and pages 7 and 8 inspected. Forty focused tests pass; supplement
TeX and PDF are unchanged.

The nested calibration revision reports greater original risk support but
no clear utility after most switches are removed. The manuscript remains
eight pages; pages 1 through 6 are identical to the last render and pages
7 and 8 were inspected. Forty-three focused tests pass. Supplement TeX
and PDF remain unchanged. Independent confirmation is still unrun.

The selected-harm revision adds the matched training comparison and the
screened versus full-bank feasibility limits, with no model promotion.
The revised eight-page PDF has pages 1 through 6 and 8 identical to the
prior render; page 7 was inspected. Forty-nine focused tests pass.
Supplement TeX and PDF are unchanged.

The chronological ledger revision adds two positive matched utility
contrasts alongside the failed original risk results, including the
reoptimized physical-only comparator. The eight-page PDF has pages 1
through 6 identical to the previous render and pages 7 and 8 inspected.
Fifty-two focused tests pass. Supplement TeX and PDF are unchanged.

The credit transport revision records realized net deficits on predicted
earning queries and the donor discount that collapses to no carry. The
standalone ledger bundle is aggregate-only. The eight-page PDF retains
pages 1 through 6 exactly; pages 7 and 8 were inspected. Fifty-seven
focused tests pass. Supplement TeX and PDF are unchanged.

## Evaluation Supplement

`supplementary_math.md` and its generated TeX give the exact scope of the
registered row-support completion envelope. Partially labelled rows retain
their observed-step loss; only wholly unknown row contrasts are bounded, with
the known CV reference denominator fixed. This is not a full-grid ADE bound.
A separately documented existing full-grid calculation bounds every missing
step. Elementary proofs, attainability examples and synthetic tests keep these
quantities distinct without changing any registered metric or historical score.

The synthetic checks and manuscript checks form part of the 43-test verified
suite. The three-page supplementary PDF compiles and every rendered page was
inspected. All 15 active registered model/protocol dependency hashes remain
unchanged. The dual-view disk interruption recovered for exactly the remaining
65 updates in the tenth fit, with final RNG/draw parity; that engineering
success is separate from the failed predictive comparisons.

```sh
.venv-pytorch/bin/python scripts/build_m3w_native_supplement.py
python3 /Users/yangyue/.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py \
  data/stage_cvpr2027_experiments/evidence_manuscript_v10/supplement_build/main.tex \
  --compiler tectonic --json
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_bound_estimands.py tests/test_m3w_native_manuscript.py
```
