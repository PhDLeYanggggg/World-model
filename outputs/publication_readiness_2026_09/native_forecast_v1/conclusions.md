# Native-Loss Training Improves Forecasts, Not Yet Deployment Safety

## Decision and Result

The delegated metric choice has led to a completed controlled training experiment,
not just a favorable rescore. Both loss arms use the same causal Transformer,
initialization, training batches, optimizer, schedule and update budget. Native
loss weighting improves forecasting in all four explored source scenes and all
three seeds. It also causes larger errors on some perfectly CV-predictable paths.
This is a promising predictor for subsequent risk-learning experiments, not a
safe deployed model or a submission-ready contribution.

Result source: **fresh_run** for 24 real Torch fits and held-source inference;
**cached_verified** for existing source inputs, identities and earlier causal
baseline controls. Independent confirmation, native-risk calibration, new
deployment and external transfer are **not_run** in this experiment.

## Fixed Comparison

The registration was committed as `c6c25a4d` before fitting. There are four
previously explored SDD auxiliary training sites, 33 recordings and 175,756
past-eligible queries. Each fit trains on three physical sites and predicts the
excluded fourth. No original validation/test, main, external or bookstore
readout is opened. These are source-exclusion development results, not a restored
independent test after earlier exploration.

Eight observed annotation steps predict twelve steps. The SDD cache uses stride
12, so the last requested endpoint is +144 raw annotation frames. This does not
establish seconds, metric scale or standard physical-time benchmark equivalence.
Raw-frame t+50 remains a separate supplement, not the endpoint scored here.

The existing history/neighbor Transformer has 88,514 parameters. Neither arm uses
images, JEPA, learned goal clusters or a pretrained teacher. A motion-bound makes
exactly static ego histories return CV; missed static starts remain evaluated.
The comparison isolates a loss weighting, not a new multimodal architecture or
proof that neighbors contribute. Inputs follow the approved offline supplied
annotation contract, not a strict real-time sensor-as-of reconstruction.

All 24 endpoints finish at 4,000 updates: 96,000 updates and 6,144,000 sampled row
draws. Actual unique-row coverage per fit is 73.35%-93.92% of the eligible training
complement, not a complete pass through every row. The two arms have identical
draw counts and final sampler states for each site/seed pair. The complete
[training trace and coverage](training_diagnostics.md) is retained.

## Forecasting Evidence

All gains below use the same native-coordinate scoring. Means average per-seed
errors, not predictions from a new ensemble. Within each physical scene, errors
are compared to causal CV, then the four scene-relative gains are averaged.
CV was also selected as the strongest fixed candidate in every complementary
source-site selection in the preceding hash-bound baseline audit.

| Loss Objective | Seed 17 ADE Gain | Seed 29 | Seed 43 | Mean ADE Gain | Mean FDE Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old past-normalized loss | 2.2144% | 2.1396% | 2.1958% | 2.1833% | 2.7966% |
| Native-coordinate loss | 7.7259% | 7.8749% | 7.2984% | 7.6331% | 8.6451% |

The native model's mean ADE scene-bootstrap interval is **[5.9632%, 9.3022%]**.
Direct matched error reduction versus the old-loss model is **5.5583%**, interval
**[3.3401%, 7.7766%]**. This is a paired relative-error contrast, not subtraction
of the two aggregate gain percentages. Across-seed ADE-gain standard deviation
is 0.2992 percentage points for native and 0.0389 for old loss.

| Physical Site | Supported Queries | CV ADE (px) | Old-Loss ADE (px) | Native-Loss ADE (px) | Native Gain vs CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa | 27,778 | 11.9082 | 11.7524 | 10.8785 | 8.6472% |
| deathCircle | 35,760 | 24.8107 | 24.0661 | 23.5531 | 5.0686% |
| gates | 20,236 | 19.7615 | 19.2831 | 18.4060 | 6.8594% |
| hyang | 89,183 | 15.9234 | 15.6044 | 14.3379 | 9.9571% |

Supported masked ADE includes 172,957 queries; 2,799 have no future support and
remain unknown. Complete-future sensitivity covers 143,918 queries: mean native
gain is 7.9254% versus 2.1880% for old loss. Endpoint-supported FDE covers 144,010
queries. All masks and row identities are paired rather than independently
filtering favorable outcomes for each model.

The prespecified training-complement-q75 CV-hard diagnostic has 41,844 queries.
Its three native seed gains are 10.6913%, 10.7742% and 10.5070%, mean **10.6575%**;
old-loss mean is 3.7363%. This descriptive slice is not a replacement for an
independently registered and calibrated hard/failure deployment gate.

Every mean-seed scene p95 ADE improves over both CV and the old-loss model.
The complete per-seed scene means, p95/p99 errors and named slices remain in
[analysis.json](analysis.json); no seed, site or checkpoint is selected away.
The 3,000 bootstrap draws resample four physical scenes, not overlapping windows.
With only four explored sites and overlapping training complements, those
intervals are conditional descriptive uncertainty, not independent confirmation.

## Safety Failure Must Remain Visible

Exactly static-history predictions match CV, giving zero improvement on that
slice. This protects static stays but cannot recover static-to-moving starts.
It does not protect every easy trajectory: nonstatic paths can also be perfectly
predicted by CV. On 11,566 complete queries with exactly zero CV error, the neural
model introduces positive error. Percent degradation has a zero denominator;
it is undefined, not a zero-degradation pass.

| Physical Site | Zero-CV Queries | Old-Loss Mean Absolute Harm (px) | Native-Loss Mean Absolute Harm (px) |
| --- | ---: | ---: | ---: |
| coupa | 2,463 | 0.017153 | 0.344945 |
| deathCircle | 2,984 | 0.001006 | 0.918078 |
| gates | 1,565 | 0.001565 | 1.361339 |
| hyang | 4,554 | 0.004254 | 0.511950 |

These values average all three seed errors within each scene. The full native
easy-preservation gate is **not certified**: its risk definition and independent
calibration have not been registered under the amended metric. The observed
zero-CV harm already rules out claiming that this unrestricted predictor preserves
all easy cases. No threshold is tuned on these readouts and no deployment changes.

The old normalized scores are also retained. Native-loss gains on that old metric
are -0.000908%, -0.000988% and -0.002082%; old-loss gains are about +0.00175%.
The new model does not retroactively solve the old static-start-weighted task.
Old negative experiments and the earlier static-subset readout remain valid on
their stated populations. Neither comparison should be mixed with historical
Stage26/37 numbers from different, partly exposed protocols.

## What to Pursue Next

The evidence supports keeping the native-loss candidate for research, without
selecting a lucky seed or enlarging the model immediately. The next discriminating
experiment is clean, nested baseline-relative gain/harm learning and selective
intervention against this frozen candidate. Its producers, head preprocessing,
model selection and risk calibration must exclude the evaluated scene recursively.
Simply splitting an existing OOF cache again would reintroduce upstream exposure.

Before a deployment claim, register how native easy-case risk handles zero-CV
references, retain absolute harm and tail reporting, and evaluate on independent
calibration/confirmation roles. Compare no gating, independent agent gating and
matched scene-joint gating with the same predictor and intervention budget.
This comparison has not yet been run. Loss reweighting alone is not the proposed
novel method; robust conditional risk and useful joint decisions still need proof.

## Verification and Boundaries

All required processes exited successfully. Exact cached metric replay passes.
A separate scalar-error implementation checks 1,054,536 query predictions and
188 scene reductions; reloading all 24 checkpoints reproduces 7,752 fixed
inference rows exactly. Twelve paired samplers match, all 24 output heads change
from initialization, and no held-site row is sampled for training. These checks
are engineering/computational verification, not a second independent study.

39 scoped tests pass. The full historical, report-writing test suite was not
rerun. Native arm64 Torch CPU uses four compute threads, one interop thread and
zero DataLoader workers. Summed fitting time is 33.40 minutes; evaluation and
verification are additional. No CREATE job or new remote availability claim.

Analysis SHA256:
`e685f0afc94149a5bc2fe231cabacea6ca108a0e52843546a1519dbc6a4198b3`.
There are 217 bound dependencies. See [replay verification](verification_with_replay.json)
and [execution notes](execution_notes.md). Checkpoints and query-level artifacts
stay local and are not GitHub deliverables.

SDD remains pixel/raw-frame, not metric/seconds. The project is not true 3D or a
foundation world model. No cross-dataset success, Stage5C execution, SMC, new
deployment or submission-ready claim follows from this experiment.
