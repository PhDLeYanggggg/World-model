# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
keep the cases that the baseline already handles well from getting worse.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal trajectory forecasting, learning intervention gain and harm,
and deciding when a neural prediction should replace a physical baseline.
Combining JEPA, Transformer and other modules is not, by itself, a contribution.

## Current Result

I completed nine fixed-budget EqMotion forecasting fits on the EuropeanSquares
SOURCE data, using three real training seeds and the same observations, labels
and training exposure as the registered PCT controls. This is a fixed-head K=1
adaptation of an existing public method, not a new architecture or reproduction
of published best-of-20 scores.

On observed labels, EqMotion improves available-label ADE relative to CV by
17.21% on the original SOURCE population. The stronger PCT controls matter:

|Comparator|SOURCE target windows|EqMotion minus control, known gain pp [95% CI]|
|---|---:|---:|
|Original PCT, registered primary|318,969|+14.52 [+9.55, +22.19]|
|Dimensionless agent-track PCT|318,969|+8.75 [+6.09, +11.93]|
|Current history-detector PCT|149,510|+4.68 [+1.39, +7.41]|

The last comparison uses the unchanged controller whitelist: EqMotion gains
10.81% versus CV there, compared with 6.14% for current PCT. The two populations
are not pooled. Differences are percentage points using a common known CV
denominator within locality, not percent error reduction relative to each PCT.
Intervals use 3,000 locality resamples after producer-fold and seed averaging.

All prescribed forecasts and 216 primary/secondary metric groups replay.
However, these are exposed SOURCE crossfit results, not independent final tests.
All missing-outcome lower intervals cross zero, and locality 082 loses to the
current PCT on known labels. **I have not upgraded deployment or claimed a new
safe world model from this result.** It provides a stronger proposal for the
next admission experiment.

![Observed SOURCE gains and unresolved missing outcomes](outputs/publication_readiness_2026_09/european_eqmotion_v1/figures/strong_controls.svg)

[Complete forecasting comparison](outputs/publication_readiness_2026_09/european_eqmotion_v1/readout.md),
[stronger controls](outputs/publication_readiness_2026_09/european_eqmotion_v1/strong_controls/readout.md),
[model/data card](outputs/publication_readiness_2026_09/european_eqmotion_v1/model_data_card.md)
and [failure analysis](outputs/publication_readiness_2026_09/european_eqmotion_v1/failure_analysis.md).

## What Has Not Worked

Better forecasting, better cost fitting and safe admission are different tests.
The existing controller improves average observed error but does not satisfy
the complete recording-level risk requirements. Recent controlled experiments narrow
the explanation without turning it into a success:

- Eighteen matched cost forests fail the registered paired lower contrast and
  complete risk contract. Reference sharing and projection slightly help on
  SOURCE; the fitting gap is already present before the decoder.
- Doubling the neural critic width reduces its full SOURCE fitting objective
  by only 0.43%. Its registered lower contrast is -0.636 pp, and only 9/252
  required recording/pipeline views pass. It is not deployed.
- The completed 102-recording calibration experiment finds no common risk
  multiplier that satisfies the original complete contract. Empty selected
  reference support is not counted as a measured safe zero.
- Explicit observation-support cost models reduce some risk violations but
  lose 2.56 pp of observed utility against the frozen direct head. Their full
  risk check still fails; better label-availability prediction is not a policy win.
- A complete, capacity-matched interaction-feature test lowers the fitting
  objective by about 11.5%, but its held-scene primary remains inconclusive:
  -0.021 pp [-0.615, +0.648]. Only 110/444 repeated recording/pipeline views
  meet the complete risk contract. Better latent features are not yet a safe
  admission result.

The current contract retains the 2% all/easy positive-harm and signed easy-net
requirements in every required recording, with finite support. A favorable
global mean or one easy-error statistic does not replace that contract.

The [results ledger](README_RESULTS.md) contains the full numerical record.
The [previous front page](README_RESEARCH_HISTORY_2026_11_03.md) is archived
verbatim; its pending states and proposed next steps are historical, not current.

## Protocol And Limits

The main observation protocol is **8 observed steps and 12 requested steps**.
The European-source experiment uses a stride of 12 raw frames. Historical
raw-frame `t+50` scores are a separate task and are not mixed into these results.

Future targets are supervision and evaluation only. Future endpoints, central
velocity, test-endpoint goals, future-validity masks and test normalization do
not enter inference. Producer, preprocessing and teacher exposure follow the
same data-role boundaries. Calibration private arrays, final confirmation and
DroneCrowd remain closed for the current experiments.

The project remains a **2.5D multi-agent trajectory/world-state research system**,
not true 3D or a foundation world model. The new forecasting control is 2D
image-plane geometry with time, not depth. Pixel/dataset-local coordinates are
not metric without verification; raw frames are not seconds. Detector/tracker,
inferred and self-audited labels are not human gold. Harm diagnostics are not
physical-safety guarantees. Stage5C execution and SMC remain disabled.

Earlier Stage35/37/43/44 results have recording, teacher-exposure or selection
limitations. They remain exploratory; new split names do not make them
independent evidence. The stronger SOURCE forecaster is a positive component
result, not yet a submission-ready M3W method.

## Running And Reading

Use native arm64 `.venv-pytorch` on Apple Silicon, not x86 Conda under Rosetta.
The completed study uses four CPU compute threads, one inter-op thread and no
DataLoader workers, with atomic checkpoints, heartbeat and resume. It completes
36,000 primary updates, a 4,000-update full refit and a 200-update recovery pilot.
Training/refit takes 3.58 h; complete primary inference replay takes 2.10 h.
These are real Torch runs, not NumPy fallback. MPS success is not claimed.

Public scalar reports can be rebuilt without rereading raw label arrays:

```sh
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion.py
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion_controls.py
```

The full pipeline entry points are in the experiment protocol/readout. Completed
stage receipts are immutable: resume incomplete work and do not delete results
to manufacture a fresh run. Full execution still requires verified upstream
data and private checkpoints; anonymous cold-start reproduction remains a gap.
The focused suite has 75 passing tests and one optional Metal-parity skip.
The unrelated legacy suite is not presented as a clean read-only smoke test.

|Record|Purpose|
|---|---|
|[Results ledger](README_RESULTS.md)|Detailed outcomes and negative results|
|[Research state](research_state.json)|Current status and next action|
|[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v9/README.md)|Evidence draft, not submission-ready|
|[Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md)|Role and exposure boundaries|
|[Local/CREATE guide](outputs/publication_readiness_2026_09/local_create_runbook_zh.md)|Runtime and recovery guidance|
|[Historical lineage audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md)|Limits of earlier claims|

Raw datasets, feature/history/latent caches, large checkpoints, videos,
third-party images and local virtual environments stay out of Git.

## Next Question

Can the stronger frozen proposal support useful, reliable admission rather
than another increase in critic width? The completed SOURCE experiments
narrow that question. The negative 082 slice and missing futures stay in the
analysis; none of these results changes deployment.

The [first amplitude experiment](outputs/publication_readiness_2026_09/european_eqmotion_amplitude_v1/readout.md)
is complete: all 1,440 policy evaluations replay exactly. The prespecified
quarter-amplitude EqMotion gains 6.28% over the fixed causal floor, but 0/444
recording views satisfy the full risk contract. Fixed support removes undefined
reference denominators; it does not remove actual easy harm or uncertainty from
missing futures. Even amplitude .001 fails six views of the same recording,
while gaining only 0.027%. No grid point is chosen for deployment.

The [causal label-support probe](outputs/publication_readiness_2026_09/european_source_label_support_v1/readout.md)
is also complete. On excluded SOURCE sites, past detector history raises AUROC
from 0.602 to 0.706; the paired feature effect is +0.104 [0.060, 0.162]. AUPRC
is 0.144 versus equal-locality unknown prevalence 0.042. Eighteen full forests,
real recovery/full refit, all probability arrays and 144 metric groups replay.
Training takes 267.97 s with 4.51 GB peak RSS. Future masks are targets only.
Calibration remains weak on the critical scene, and some scenes do not improve.
This is a release-label-process result, not physical visibility or neural dynamics.

The [matched bound-aware cost experiment](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/readout.md)
is complete: 36 real Torch heads, 116,160 primary updates, 4,000 full-refit and
1,200 recovery-pilot updates. Both fitting stages exclude each evaluation site;
all cost/amplitude arrays and 72 metric groups replay exactly. Training takes
378.71 s with 9.24 GB peak RSS. Explicit unknown-row targets lower their own
fitting objective from 4.884 to 1.489, but the direct paired primary is
**-1.70 pp [-5.71, +1.12]**, not a supported improvement. Known utility differs
by +0.38 pp [-1.48, +1.95], also inconclusive.

Full-risk support improves from 28/444 to 108/444 repeated recording views,
while known-all violations increase. Neither policy passes. In the worst
signed easy-net view, predicted easy risk is 0.060% but observed easy positive
harm is 57.76%. The [failure analysis](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/failure_analysis.md)
separates target-fitting progress from this selected-risk failure. I will next
decompose harm and reference errors on identical selected sets, on fitting and
excluded SOURCE sites, before choosing another cost-learning repair. No new
threshold will be selected from exposed calibration or final-confirmation data.

The [same-set cost diagnostic](outputs/publication_readiness_2026_09/european_bound_cost_decomposition_v1/readout.md)
is also complete, with full frozen-model replay. It locates failure before
new-scene transfer: on fitting sites, the bound-aware head's median selected
EH predicted/target mass ratio is only 0.0065. Every one of its 358 falsely-safe
fitting views underestimates EH; only 40 also inflate ER. Error is therefore
not just a denominator problem. Selection worsens the unattenuated EH mass
ratio in 423/444 fitting views. Fitting views with no wholly unknown selected
labels can also fail, so missing-label bounds alone do not explain it. This is a zero-update diagnosis, not a
new policy or another training success.

The [observation-support comparison](outputs/publication_readiness_2026_09/european_observation_cost_v1/readout.md)
is complete: 36 real Torch fits, 116,160 primary updates and exact full replay.
Training takes 404.41 s with 9.23 GB peak RSS. Its primary paired lower effect
is **-3.13 pp [-4.41, -1.99]**; observed utility also regresses,
**-2.56 pp [-3.74, -1.48]**. All twelve locality-averaged known effects are
negative. Risk support improves to 160/444, but worst easy-net degradation is
8.32%, so this model is not deployed.

The identified availability head learns more about released labels, but average
intervention amplitude falls from 0.239 to 0.091 and useful gains are lost.
The [failure analysis](outputs/publication_readiness_2026_09/european_observation_cost_v1/failure_analysis.md)
keeps that distinction explicit. The subsequent
[frozen interaction-feature experiment](outputs/publication_readiness_2026_09/european_interaction_feature_cost_v1/readout.md)
is also complete: 36 real cost-head fits, three actual seeds, exact replay of
all predictions and 72 metric groups. The two-hour feature extraction and
separate two-hour re-extraction preserve every original forecast. The added
feature bundle improves fitting, but its registered lower effect and known
effect both have intervals crossing zero. It is not a new safe policy.

The [new same-set diagnosis](outputs/publication_readiness_2026_09/european_feature_cost_diagnosis_v1/conclusions.md)
finds that this head underestimates selected easy harm already on fitting
scenes, not only on new localities. All 324 falsely-safe fitting views and 330
held views underestimate its numerator. I then tested whether more training
could fix this before changing the architecture again. The
[complete fourfold-budget experiment](outputs/publication_readiness_2026_09/european_feature_duration_probe_v1/conclusions.md)
finishes all 18 cost heads, 232,320 primary updates and a separate exact replay.
Average fitting loss improves 34.77%, but selected easy harm remains badly
underestimated. On the unchanged original decisions its prediction/target
ratio rises from0.0032 to0.0093; on the new model's own decisions it is0.0028.
The latter still violates easy risk in 307/444 repeated recording/pipeline
views. Longer training alone has not produced a safe learned policy.

The run recovered from a disk-reserve interruption without lowering the
budget or discarding completed heads. Both the original and completed
checkpoint sets are verified backups on CREATE; no remote training was
needed. The completed
[label-support decomposition](outputs/publication_readiness_2026_09/european_duration_support_diagnosis_v1/conclusions.md)
shows why missing labels are not the whole explanation. Unknown-row bounds
account for a median 64.26% of the evaluated easy-harm numerator, but 223 views also fail on
available labels. Even on fully labeled twelve-step paths, predicted easy
harm is only 0.83% of the target. Neither lower loss, fewer interventions nor
dropping uncertain labels establishes safe world-dynamics improvement.

I am now checking a different development-data route:
[MovingDroneCrowd++](outputs/publication_readiness_2026_09/movingdronecrowd_intake_v1/conclusions.md),
which provides image sequences and head-identity annotations. Its version is
pinned before acquisition. The completed annotation pass finds 120 clips,
638,718 labeled rows and 245,018 complete eight-observation/twelve-prediction
windows, using 7.18 MB of network payload rather than the full archive.
These are overlapping native-step windows, not independent samples or matched
physical time. The existing DroneCrowd collection remains
reserved for confirmation in its entirety. The new source is not yet admitted
for training or claimed to be independent of those reserved scenes. The next
step checks source overlap before assigning research roles.

The [registered image-overlap study](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/protocol.md)
covers every new clip and its relation to the protected DroneCrowd collection.
It keeps ambiguous matches as exclusion constraints. Passing a software check
or finding no image match will not be treated as proof of a new physical site.
Acquisition exposed 21 portrait views that need an explicit orientation fix
before matching; the original pass is retained and no training role is granted.

![Bound-aware cost fitting does not establish full policy safety](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/figures/cost_target_comparison.svg)

The larger goal remains useful neural dynamics backed by reproducible,
independent evidence. More stages or a lower training loss are not substitutes.
