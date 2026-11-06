# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
keep the cases that the baseline already handles well from getting worse.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal trajectory forecasting, learning intervention gain and harm,
and deciding when a neural prediction should replace a physical baseline.
Combining JEPA, Transformer and other modules is not, by itself, a contribution.

## Current Result

The complete-policy evaluation is now finished. I reconstructed all 62,466
past-eligible rows in eight exposed development sources, including unknown
labels and non-interventions, and kept the original per-record 2% risk and
support requirements. No new held test, training or threshold search was used.

**Trajectory accuracy improves, but neither consensus policy is safe to deploy.**

The fixed mean rule improves available-label source-macro ADE by 1.313%, and
unanimity by 0.336%, versus the training-defined motion floor. Conservative
completion lower gains are 0.824% and 0.306%. Yet only 1/24 and 2/24 required
recording views pass the full original risk/support contract. This is not just
an empty-denominator problem: there are 15 and 9 known-easy positive-harm
violations, respectively. Unknown labels add further conservative risk.

All recordings have easy observations before selection, but consensus removes
much of the selected easy reference mass. The next repair targets complete-risk
training and reference support, not a looser safety rule. Empty support stays
not estimable; there is no identity-fallback exemption. Deployment is unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/operation_zh.md).

### Risk-Head Seed Replication

The donor benefit now replicates across risk-head seeds 17/29/43 with the
forecast teacher fixed. I trained 48 additional heads and repeated every new
fit exactly, retaining the verified seed17 models rather than selecting a
favorable run. The information criterion passes in 3/3, 2/3 and 2/3 contexts.

**The component learns better; the subsequent complete-policy check fails.**

Average individual-head decision cost falls 12.16% versus matched own-only
training, with an exploratory source-component interval of [2.66%, 27.18%].
The forest comparison is 4.15% [-4.75%, 8.12%], which does not establish a
reliable advantage over that stronger control. One seed/context has worse MSE.
These intervals use eight exposed source components, not overlapping windows,
and do not describe the separately aggregated score ensembles.

The fixed three-head unanimity rule reduces harmful known-proposal admissions
to 7/26/26, versus 46/229/602 for the original single head and 0/1/0 for the
forest. It also rejects useful predictions. These component counts exclude the
unknown-label contributions and reference/support conditions evaluated in the
new complete-policy result above. The outer test stays closed. Deployment and
the original rules remain unchanged.

These are three risk-head seeds, not three forecaster seeds or independent
world-model confirmation. The project is not submission-ready.
[Replication and uncertainty](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/README.md),
[fixed consensus](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/consensus/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/failure_analysis.md).

### Previous Seed17 Donor Comparison

The new source-held donors improve neural risk learning in the matched
development comparison. I trained 24 heads and repeated all 24 fits exactly
before evaluation. Each receives the same 4,096-update budget and original
causal features; the comparison is own-B only, producer-A only, and a 50/50 mixture.

**The mixture improves information, but is not yet safe to deploy.**

Decision cost falls 20.71%, 32.27% and 10.69% versus matched own-only training;
risk MSE also falls in all three contexts. All eight source folds have lower
total decision cost. This supports the donor treatment rather than another
blind increase in training time. It does not isolate source count from teacher
size and candidate-mixture effects, and remains a one-seed development result.

Against the more conservative forest, total cost improves in two contexts but
worsens in one, and false-safe cost is higher in all three. Source060 also has
more harmful admissions than own-only despite its lower total cost. The 3/3
information screen is not a safety-gate pass. Deployment, the original 2% rules,
empty-support NA and the closed outer test remain unchanged.
[Matched results](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/operation_zh.md).

### Source-Held Forecast Donor Bank

I trained six new EqMotion controls to expand risk-learning source coverage
without relaxing the safety rule. Each completed 4,000 updates. Their forecasts
are held out by source inside the original predictor training sets, keeping
outer calibration/evaluation roles closed. The bank now covers 125,540 rows
from 6 and 8 sources in the two separate parent training sets.

**The training-data repair is complete; safer intervention is not yet proved.**

All six forecast caches replay exactly. Donor ADE beats causal CV on 10 of 14
sources and loses on four; all sources and unknown-label rows remain in the
record. These are training-donor diagnostics, not a new independent test or a
strongest-baseline result. The matched donor comparison and its risk-head seed
replication are recorded above. Deployment and the original 2% rules
remain unchanged; the paper is not submission-ready.
[Training and donor results](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/README.md),
[source eligibility](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/eligibility_report.md),
[reproduction](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/operation_zh.md).

### Previous Budget-Controlled Risk Experiment

I tested whether the neural risk failure was simply due to too little training.
I continued all eight saved heads, adding 32,871 updates with the same model,
features, loss, sources and original risk rules. Only the fixed final checkpoint
was evaluated after every fit and exact repeated continuation had completed.

**More training improved fitting, but did not repair transfer.**

Fitting loss fell in all eight folds; source-held risk MSE worsened in six.
The decision-cost reductions versus the forest control are -4.72%, -68.65%
and -6.69% across the three development contexts. The screen passes 0/3, so
deployment is unchanged. These are risk-model results, not trajectory gains.

The fitting banks each contain only one or two source components. I am now
prioritizing compatible source coverage and conditional risk learning over
another unstructured increase in training time. The original 2% and support
requirements remain; empty support is NA. No new reserved test was opened.
The paper is not submission-ready.
[Results and loss traces](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/failure_analysis.md),
[source coverage](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/source_coverage.md).

### Previous Neural Risk Result

I trained a small neural risk model to adjust the mismatch between a query
and its reference histories. Plain neural prediction and a linear correction
serve as controls; the reference set, causal inputs and original rules stay fixed.

**The networks trained and reproduced, but did not improve safe intervention.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | -42.675% | -6.914% | 0 to 94 |
| 1 | -18.759% | -54.522% | 1 to 353 |
| 2 | +32.192% | -5.517% | 0 to 1288 |

Eight Torch networks and eight ridge controls complete; every fitting row gets
32 epochs. All models reproduce by full refitting. Correcting reference risk
helps modestly versus the plain network, but not versus the forest control.
The information screen passes 0/3 contexts. Actual training losses are retained, and
convergence is not claimed. This is risk-model development, not new neural
trajectory success, independent confirmation or a safety certificate.
The original 2% and support requirements remain, empty references stay NA,
and deployment is unchanged. The paper is not submission-ready.
[Results and loss traces](outputs/publication_readiness_2026_09/moving_neural_transport_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_neural_transport_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_neural_transport_v1/operation_zh.md).

### Previous History-Matching Result

I tested whether similar causal histories from other sources can identify
useful interventions that risk forests miss. The new local risk model uses
32 source/agent buckets per query and excludes the query's entire source.
It leaves the features, targets and original safety rules unchanged.

**Matching history carries information, but still does not make intervention safe.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | -0.427% | -4.482% | 0 to 107 |
| 1 | -0.199% | -121.691% | 1 to 723 |
| 2 | +33.420% | -2.568% | 0 to 521 |

Correct matching beats a fixed misaligned control, but loses to the stronger
forest on decision cost in all three settings. The fixed information screen
fails 0/3. All 48,061 known proposals were processed, and all eight complete
retrievals reproduce exactly. These are exposed development component results,
not trajectory gains, independent test evidence or a safety certificate.
Deployment stays unchanged. The original 2% and support requirements remain;
empty reference support is not estimable, not passed. No identity exemption.
The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_track_analogue_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_track_analogue_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_track_analogue_v1/operation_zh.md).

### Previous Channel-Head Result

I traced why recent risk-regression improvements did not improve intervention
decisions. Almost all of the error reduction occurred on proposals that both
models still rejected. I then tested a concrete model change: separate
all-agent and easy-agent risk regressors, keeping the total tree budget,
causal features, targets and safety rules unchanged.

**Separating the heads did not repair safe intervention.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +2.909% | -0.034% | 0 to 0 |
| 1 | +0.372% | +0.461% | 1 to 19 |
| 2 | -3.898% | 0.000% | 0 to 0 |

The fixed information screen fails 0/3. Sixteen new model objects and eight
verified controls reproduce by full refit; that establishes reproducibility,
not effectiveness. These are exposed source-held component diagnostics, not
trajectory improvements or an independent test. I keep the current deployment
unchanged and retain the original 2% risk and support requirements. Empty
reference support remains not estimable. The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/operation_zh.md).

### Previous Motion-Frame Result

I completed a motion-frame repair suggested by the tail audit: express motion
in the agent's own history scale and retain the original scale context.
Sixteen new models and eight verified controls keep the same signed targets,
risk requirements and data roles. All models reproduce by full refit.

**Better risk regression still does not establish better safe decisions.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +9.825% | +0.115% | 0 to 0 |
| 1 | +0.709% | +0.054% | 1 to 6 |
| 2 | +30.216% | 0.000% | 0 to 0 |

The fixed information screen fails 0/3. In context 2, all 38,579 known
proposals remain rejected. These are source-held risk-model diagnostics,
not trajectory improvements, independent confirmation or safety certificates.
No model is promoted. The original 2% and support rules remain unchanged;
empty controls stay not estimable. The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_ego_frame_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_ego_frame_v1/failure_analysis.md),
[Chinese summary of both repairs](outputs/publication_readiness_2026_09/moving_ego_frame_v1/model_repair_summary_zh.md).

### Completed Forecast-Tail Test

I trained eight positive-cost risk probes to trace the harmful tail of the
frozen forecast bank, against eight verified signed-risk controls. All models
reproduce by full refit. The original safety rules and reserved tests stay unchanged.

**Harm is concentrated, but causal tail identification still transfers poorly.**
At a fixed10% ranking weight, easy-cost capture changes from
19.20/3.42/38.23% to20.67/4.08/39.10% across three contexts. The internal
information screen fails0/3. These are diagnostic cost-capture rates, not
trajectory gains or safety passes. Fitting-source capture is much higher than
excluded-source capture. The next controlled repair concerns the motion
representation's dependence on neighbor scale, not another threshold sweep.
No new model is deployed, and the paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/protocol.md).

### Previous Cost-Partition Result

I tested a targeted change to risk learning: fit partitions around useful and
costly decisions, while keeping the original expected-risk readout. Sixteen new
models and eight verified controls use the same inputs and capacity limits.
All models reproduce by full refit; the safety rules and decision threshold
are unchanged.

**Substantially better risk regression still did not yield a safe decision upgrade.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +10.744% | -1.830% | 0 to 6 |
| 1 | +1.042% | +0.041% | 1 to 9 |
| 2 | +35.804% | +0.284% | 0 to 3 |

Positive reductions mean better. The fixed internal follow-up screen fails 0/3.
These are source-held fitting diagnostics, not trajectory gains or independent
confirmation. I do not promote this model or open another reserved-test readout.
The next question concerns the harmful tail of the candidate forecasts: whether
the predictor itself needs repair, rather than another selector-weight sweep.
The paper is not yet submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_cost_partition_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_cost_partition_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_cost_partition_v1/protocol.md).

### Last Completed Information Test

I tested whether richer directional information about nearby agents improves
intervention decisions. The comparison adds step-by-step candidate-to-peer
geometry, including a deliberately mismatched association control, and leaves
one fitting source out at a time. All 48 fixed linear and nonlinear fits are
complete and reproduce exactly.

**Better risk prediction did not produce reliable decision gains.**

| Context | Nonlinear risk-MSE reduction | Decision-cost reduction | Outcome |
|---|---:|---:|---|
| 0 | -1.262% | -0.000449% | Worse prediction |
| 1 | +0.997% | +0.189098% | More harmful admissions |
| 2 | +5.000% | 0.000000% | All candidates still rejected |

Positive reductions mean better. The primary follow-up screen passes only 1/3,
below the registered requirement of two. In context2, the favorable error change
does not alter any of the 38,579 known candidate decisions. Context1 increases
harmful admissions from 1 to 10. I therefore do not expand this feature branch
into a new neural model or promote it for deployment.

This is an exposed-development information test, not independent confirmation
or a trajectory-dynamics improvement. Original 2% risk and support rules remain
unchanged; there is no identity-fallback exemption or new reserved-test readout.
The next question is why the risk estimate fails to distinguish useful switches
from costly mistakes, with earlier negative cost-head experiments retained.
The paper is not yet submission-ready.
[Complete results](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/protocol.md).

### Last Completed Model Experiment

I separated two model-input choices that the previous experiment changed
together: which forecast represents a peer, and whether fallback peers remain
visible. Six new neural fits and six verified existing controls complete the
four-cell comparison, with matched capacity, initialization, training and loss.

**Keeping fallback peers did not produce a reliable decision improvement.**

| Context | Risk MSE reduction | Decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +0.043% | -0.043% | +8.922% |
| 1 | -0.279% | -4.108% | -19.095% |
| 2 | -0.402% | +0.675% | +9.027% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. Retaining fallback peers partially repairs two prior failures,
but the registered primary fails the fixed internal screen in all three contexts.
No additional development or reserved-test evaluation is opened.

All six fresh fits finish 1,000 updates. Resume, legacy-engine replay, a full
repeat and all twelve cells' metrics agree exactly. The original 2% safety and
support rules remain unchanged; no identity exemption or deployment upgrade is
introduced. The next question is whether candidate-relative peer information
adds source-held-out predictive value beyond the existing pooled representation.
The paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_peer_participation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_peer_participation_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_peer_participation_v1/protocol.md).

### Last Completed Peer-Context Experiment

I completed six matched neural fits to test whether risk estimation benefits
from seeing each peer's actual proposed forecast, rather than a same-index
hypothetical forecast. Both models have the same capacity, initial predictions,
training budget and loss. The original 2% risk rules remain unchanged.

**Proposal-conditioned context did not consistently improve decisions.**

| Context | Risk MSE reduction | Decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +0.067% | +0.920% | 0.000% |
| 1 | +1.078% | -6.975% | -29.496% |
| 2 | -0.363% | -0.559% | -0.331% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. In the second context, a lower average prediction error still
leads to more costly wrong decisions. The fixed internal follow-up screen fails
0/3, so no additional development or reserved-test evaluation is opened.

All six models complete 1,000 updates; resume, a full repeat and metric replay
agree exactly. Source-level analysis suggests the context change also removes
useful background information, but that is a hypothesis, not a proven cause.
The next model comparison will separate candidate identity from excluding peers
that fall back to the baseline. No deployment upgrade or safety exemption is
introduced, and the paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_proposal_context_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_proposal_context_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_proposal_context_v1/protocol.md).

### Last Completed Experiment

I completed six matched neural continuations to test whether emphasizing the
worst training source improves intervention decisions. Both arms use the same
trained starting models, inputs, query draws and 1,000-update budget. The only
change is a fixed 25% contribution from the worst training-source loss.

**Lower worst-source loss did not produce consistently better decisions.**

| Context | Worst-source loss reduction | Total decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +2.149% | -1.465% | -5.046% |
| 1 | +6.936% | -8.233% | +31.409% |
| 2 | +1.980% | +1.160% | -22.435% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. The three worst sources improve, but the five other fitting
sources worsen. None passes the fixed internal follow-up screen, so I did not
run another development or reserved-test readout. Resume, a full refit and all
fitting metrics replay exactly; that verifies execution, not scientific success.

The original 2% risk and support requirements are unchanged. Empty support is
still not estimable, there is no identity-fallback exemption, and no model is
promoted. The next model question concerns representing the complete admission
decision more directly, rather than another source-weight or threshold sweep.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_source_robust_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_source_robust_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_source_robust_v1/protocol.md).

### Latest Model-Input Diagnostic

I tested whether explicit relations between a candidate forecast and nearby
agents explain the remaining risk errors. Sixteen fixed linear probes leave out
one training source at a time. Adding these features worsens risk-prediction MSE
by 6.29%, 0.22% and 5.12% across the three contexts; decision cost improves in
only one. This does not support expanding that feature branch. It also does not
rule out nonlinear interaction models.

This motivated the source-risk training experiment above. These are exposed
training-side diagnostics, not a
new independent test or sixteen new neural networks. The original 2% rules,
support requirements and prior failures are unchanged; there is no identity-
fallback exemption or deployment upgrade.
[Diagnostic results](outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1/failure_analysis.md).

### Previous Neural Feature Experiment

I added explicit relations between a candidate forecast and the agent's past
motion: direction, speed change, acceleration alignment and departure from the
baseline. Six matched neural models test these32features with the same loss,
initialization, training budget and original safety rules.

**The features give a small development gain, but the policy is still unsafe.**

| Policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Matched masked-feature control | 0.895% | 1.170% | 8.633% | 0/17 |
| Rollout-history relations | 0.981% | 1.303% | 8.660% | 0/17 |

The contrast is+0.0865percentage points, with missing-label completion bounds
[+0.0451,+0.1279]pp, not sampling confidence intervals. Only two of seven
sources improve; four worsen and one is unchanged. Most gain comes from one
source. Removing the decision head raises average gain but worsens maximum
easy degradation to25.726%, so that is not a deployment option.

All six fits, three original controls, a full refit and prediction/metric replay
are verified. These are already exposed obs8/pred12 image-coordinate development
results, not independent confirmation or new trajectory-dynamics training.
The original2% rules remain unchanged; no identity exemption or deployment
upgrade is introduced. The paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_rollout_history_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_rollout_history_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_rollout_history_v1/protocol.md).

### Previous Paired Decision Result

I completed nine matched neural fits to test whether learning relative scores
for useful and harmful interventions improves the decision to replace a
baseline. The comparison keeps the same inputs, initialization, query draws,
training budget and original safety rules. No threshold was retuned.

**Better ranking did not produce a safer, more useful policy.**

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Joint-decision control | 0.895% | 1.170% | 8.633% | 0/17 |
| Matched pointwise loss | 1.467% | 1.840% | 8.465% | 0/17 |
| Paired ranking loss | 0.836% | 1.129% | 8.337% | 0/17 |

The primary ranking-minus-pointwise contrast is **-0.6311 percentage points**.
Its missing-label completion bounds are [-0.8664, -0.3957] pp, not sampling
confidence intervals. Ranking improves fitting AUROC over the joint control
in all three contexts, but its decision cost is worse than the matched
pointwise arm in all three. The pointwise arm recovers more useful predictions
and increases harmful admissions; its larger mean gain does not make it safe.

These are partial-outcome results from seven already exposed development
sources, not independent confirmation or complete-roster ADE. All nine fits,
three predecessor controls, one complete refit and prediction/metric replay
are verified. Sixty-six focused tests and two completed-replay checks pass.
The read-only replay command also reproduces the real completed experiment
without changing any of its 36 then-existing public files. The resumed training took
28.6 minutes, excluding the interrupted process and downtime. No model is
promoted, no reserved scenes are opened, and no identity exemption is added.
The original 2% risk, easy and support requirements remain unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_paired_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_paired_decision_v1/failure_analysis.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_paired_decision_v1/validation_receipt.md).

### Checking The Failure Mechanism

I tested two specific explanations before extending training. Complete-fitting
gradients show risk/ranking conflict in two of three contexts, but a disposable
saved-AdamW projection improves MSE without raising decision cost in only one.
It does not support a full projection-training run. Separately, changing only
neighbor storage order changes many decisions in two contexts, but averaging
all eight fixed orders worsens fitting MSE and decision cost in all three.
This is not proof that neighbor order is a bug or that invariant models cannot
work; it rejects this particular shortcut. Both diagnostics replay exactly.

These are new own-fitting diagnostics of existing neural models, not additional
trained models, independent tests or safety passes. Original 2% rules, support
requirements and failed results remain unchanged. No identity exemption or
deployment change is introduced.
[Gradient evidence](outputs/publication_readiness_2026_09/moving_paired_gradient_v1/result_interpretation.md),
[neighbor-order evidence](outputs/publication_readiness_2026_09/moving_neighbor_order_v1/README.md).

### Previous Harm-Weight Allocation Result

I completed nine matched neural fits to test whether training should put more
weight on lower-magnitude harmful interventions. A uniform-weight control has
the same total positive coefficient mass, so the comparison tests allocation
rather than simply adding a larger penalty. The original 2% safety, reference
and support requirements remain unchanged.

**Rejecting more harmful predictions did not produce a better safe policy.**
The new loss reduces harmful admissions in every fitting partition, but also
rejects more useful predictions. Relative to the equal-mass control, total
decision cost increases in all three partitions.

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Joint-decision control | 0.895% | 1.170% | 8.633% | 0/17 |
| Uniform harm weighting | 0.881% | 1.159% | 8.633% | 0/17 |
| Low-magnitude harm floor | 0.793% | 1.046% | 8.048% | 1/17 |

The primary utility contrast is **-0.0883 percentage points**, with
missing-label completion bounds [-0.1004, -0.0762] pp. These are not sampling
confidence intervals. Results cover seven already exposed development sources
and partial outcomes, not independent test confirmation or complete-roster ADE.
One passing recording view does not satisfy the full safety requirement.
Three selected-easy views remain undefined; no identity exemption is added.

All nine fits, exact predecessor controls, a complete treatment refit, metric
replay and own-fitting error attribution are complete. Fifty-one focused tests
pass; the full historical suite was not rerun. A receipt-export error was
recovered without changing the completed checkpoint. No model is promoted,
no reserved scenes are opened, and this experiment does not train new trajectory
dynamics. The next problem is distinguishing harmful from useful interventions,
not weakening the safety rule or increasing rejection indiscriminately.
[Complete results](outputs/publication_readiness_2026_09/moving_harm_floor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_harm_floor_v1/failure_analysis.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_harm_floor_v1/validation_receipt.md).

### Previous Joint-Decision Result

I completed six matched neural fits to test a concrete loss problem: a channel
with a zero target could block a useful intervention without receiving a
signed-cost gradient. Training the complete decision reduces this rejection
cost, but also admits more harmful interventions. The original 2% safety and
support rules were not changed.

**The model improves development utility slightly, but still fails safety.**

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Previous per-channel model | 0.759% | 1.009% | 8.038% | 0/17 |
| Joint-decision model | 0.895% | 1.170% | 8.633% | 0/17 |

These are source-macro, partial-outcome results on seven already exposed
development sources, not independent test confirmation. The +0.136 percentage
point gain does not offset the safety failure. All predictions and metrics
replay exactly; that establishes reproducibility, not deployability. No model
is promoted, and no reserved test scenes were opened.

The audit also corrects an earlier explanation: training already used the
router's single proposed action. All-action averaging and opposing target
signs do not explain this failure. The remaining problem is distinguishing
useful interventions from harmful ones, not simply encouraging more switches.
[Complete experiment](outputs/publication_readiness_2026_09/moving_joint_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_joint_decision_v1/failure_analysis.md),
[corrected diagnosis](outputs/publication_readiness_2026_09/moving_decision_semantics_v1/README.md).

### Previous Per-Channel Composition Result

I trained six matched models to test whether learning the combined risk and
intervention decision improves on training the two heads independently.
The original 2% limits, conservative labels and support rules remain unchanged.

**The loss change repairs fitting, but safe generalization is still unresolved.**
All three fitting contexts pass the prespecified internal screen. Compared with
independent training, all-decision cost falls 30.94%, 14.68% and 9.89%. Yet the
gain on already exposed development scenes is only +0.013 percentage points.

| Policy | Known ADE gain over floor | Worst defined easy-net degradation | Original full risk/support |
|---|---:|---:|---:|
| Independent two-head | 0.745% | 8.116% | 0/17 |
| Composed two-head | 0.759% | 8.038% | 0/17 |

These are partial-outcome development results, not independent test confirmation.
The original constant control has greater utility but also fails safety. Neither
the average gain nor lower training cost justifies deployment. No model is promoted.

All 12,000 primary updates, a 2,000-update exact repeat, prediction replay and
error attribution are complete. The three fresh controls exactly reproduce the
previous models. Thirty-six focused tests pass; the full historical suite was
not rerun. This is neural intervention-risk research, not new trajectory dynamics.
[Complete results](outputs/publication_readiness_2026_09/moving_composed_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_composed_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_composed_risk_v1/operation_zh.md).

### Previous Separate-Representation Result

I trained nine models to test separate representations for risk magnitude and
intervention decisions, with narrow and near-capacity-matched shared controls.
The original 2% safety and support rules are unchanged.

**The change helps fitting in two contexts, but does not solve the problem.**
The separated model reduces MSE by 20.42%, 28.68% and 10.07%. Its all-decision
cost rises 4.13% in the first context and falls 30.71% and 20.18% in the other two. The failed
context admits fewer harmful predictions but rejects too many useful ones.
Separation also does not consistently outperform the wider shared control.

All 18,000 primary updates, a 2,000-update exact repeat and metric/error replay
are complete. Twenty-nine focused tests pass; the full historical suite was
not rerun. The prespecified fitting screen fails, so no additional held readout
is run and no model is promoted. This is risk-head research, not new trajectory
dynamics or independent confirmation. These findings motivated the composed
intervention loss experiment above, preserving the original rules and labels.
[Complete results](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/operation_zh.md).

### Previous Complete-Fitting Result

I tested complete-fitting optimization of the neural risk model without changing
the original 2% safety or support rules. Three real fits, exact resume checks,
a full refit and error-attribution replay are complete.

**Better loss did not produce consistently better decisions.** Mean fitting
loss falls 31.98%, 37.90% and 41.13%, but every context worsens at least one
decision-cost channel. One model rejects more useful interventions; others
admit more costly harmful ones. These are training diagnostics, not test gains.
The prospective fitting screen fails, so no new held predictions are evaluated.
There is no deployment upgrade, changed threshold or identity certification.

Twenty-seven focused checks pass; the full historical suite was not rerun.
This is native obs8/pred12 SOURCE-development evidence, not independent testing
or a new trajectory-dynamics result.

This motivated the separate-representation experiment reported above.
The earlier complete-fitting result remains negative.
[Complete result](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/README.md),
[error analysis](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/operation_zh.md).

### Previous Training-Schedule Result

I kept the original 2% safety and support rules and tested a training change.
Six matched neural risk-head fits compare a fixed learning rate with gradual
reduction, starting from identical weights and using the same inputs, labels,
loss and update budget. Full training, exact refit and prediction replay are
complete. A preceding fitting diagnostic did not support gradient conflict as
the local explanation for failure.

| Training schedule | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Constant | 3.60% | 4.63% | 38.90% | 0/17 |
| Gradually reduced | 3.75% | 4.75% | 44.65% | 0/17 |

**This is not a safe improvement.** The small average gain comes with more
easy-risk violations, and the original safety gate still fails. Full-fitting
risk errors also increase, so the experiment does not establish that the
optimization problem has been solved. I am retaining the negative result and
not deploying this model.

These are repeated development observations on seven SOURCE units and one
producer seed, not independent tests or new trajectory dynamics. No reserved
data, thresholds, support rules or identity-fallback certificates changed.
Thirty-four focused tests pass; the full historical suite was not rerun.
[Complete results](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/result_interpretation.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/reproducibility.md).

### Completed Scale-Input Ablation

I tested whether explicit pixel scale misleads the neural intervention-risk
model. Six real retrained models, a full refit and exact prediction replay
are complete. I removed only the risk head's log-scale input and retained
the original 2% safety and support requirements.

| MSE risk head | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Original input | 3.60% | 4.63% | 38.90% | 0/17 |
| Explicit scale removed | 4.15% | 5.45% | 47.59% | 0/17 |

**Average utility improves, but the safety repair fails.** The matched gain
is +0.5525 percentage points. Some scenes benefit, another loses useful
interventions, and the worst easy-case degradation increases. The secondary
signed-cost comparison shows the same aggregate utility/safety tradeoff.
Neither version is deployed, and neither passes its expansion screen.

These are repeated development results on seven SOURCE units and one producer
seed, not independent confirmation. Missing-reference and empty-control cases
remain NA. Removing one input is not full-policy coordinate invariance, and
this experiment does not train new trajectory dynamics. Evaluation rules and
reserved data are unchanged.
Thirty-one focused tests pass; the full historical suite was not rerun.
[Complete comparison](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/result_interpretation.md),
[running and recovery](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/operation_zh.md).

### Completed Signed-Cost Comparison

I tested a decision-aligned risk objective that charges mistakes according to
their realized cost. Six real neural fits, a complete treatment refit and exact
prediction replays are complete. The original 2% and support rules are unchanged.

| Risk objective | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Ordinary MSE | 3.60% | 4.63% | 38.90% | 0/17 |
| Signed-cost logistic | 2.69% | 3.56% | 26.23% | 1/17 |

**Better fitting does not establish safe transfer.** Training easy-channel
decision cost falls 81-85%, but held-source utility declines. Three sources
improve and four worsen; useful intervention nearly disappears in some scenes.
The one passing view contains only 13 selected rows. This is not an overall
safety pass or a reason to deploy the new model.

These are repeated development results on seven SOURCE units and one producer
seed, not independent confirmation or new world dynamics. Missing supports and
empty utility controls remain NA. No reserved data were opened or evaluation
rules revised. Twenty-six focused tests pass; deployment remains unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_signed_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_signed_decision_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_signed_decision_v1/operation_zh.md).

### Completed Source-Held Error Test

I tested whether errors discovered on an unseen fitting source teach a better
intervention-risk model than errors discovered on an already seen source.
Eight discovery networks were trained from scratch, followed by six matched
risk-head continuations. All 14 models completed training, and full refits and
prediction replays reproduce exactly. The original 2% rules remain unchanged.

| Error discovery | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| In-source control | 3.20% | 4.04% | 25.00% | 0/17 |
| Source-held errors | 2.95% | 4.06% | 64.13% | 0/17 |

**This repair does not establish safe utility.** The known paired contrast is
-0.2505 percentage points. Six sources improve, but a 5.03-point loss in the
seventh outweighs those gains. The method rejects many beneficial interventions
in one source while admitting beneficial but easy-harming interventions in
another. Even on the full fitting data, false-safe easy signs increase relative
to the control despite lower loss on the emphasized errors.

This is a development result on seven exposed SOURCE units and one producer
seed, not an independent test or a new world-dynamics contribution. No reserved
data were opened. Missing-support cases remain NA; no identity certificate,
deployment change or submission-readiness upgrade. Twenty-one focused tests
pass. I retain the negative result and the pre-training cache-interface repair.
[Full results](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/README.md),
[mechanism and limitations](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/result_interpretation.md),
[original-rule gates](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/gates.md).

### Completed Fixed-Error Priority Test

I tested whether teaching the risk model to correct its earlier mistakes
improves safe intervention. Both models continue from identical weights;
one uses ordinary MSE, while the other gives its fixed, observed training
errors ten times the weight. The original 2% safety and support rules stay
unchanged. Six real neural fits, a full refit and exact replay are complete.

| Risk model | Known ADE gain over floor | Hard gain | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|---:|
| Warm kinematic model | 3.94% | 4.86% | 17.32% | 0/17 |
| Ordinary continued training | 3.60% | 4.63% | 15.01% | 0/17 |
| Fixed error-priority training | 3.03% | 3.85% | 13.59% | 0/17 |

**This objective repair fails.** The emphasized fitting-error loss falls
57.96-89.21% below the matched control, but held-source utility decreases.
The model learns many old mistakes, introduces new ones, and rejects useful
interventions. Both arms retain 14 all-risk and 13 easy-risk violations,
with two undefined selected-easy views. Worst defined easy-net degradation
is 22.67% for priority training, still far above the original requirement.

This is a controlled negative result on seven exposed SOURCE units and one
producer seed, not independent confirmation. Each fitting context has only
two or three source groups; many overlapping windows cannot substitute for
scene diversity. Neither more identical training nor this fixed error
weighting establishes safe utility. I have not changed the evaluation rules,
certified missing support, upgraded deployment or claimed a new dynamics method.
[Complete results](outputs/publication_readiness_2026_09/moving_admission_risk_v1/README.md)
and [what the experiment identifies](outputs/publication_readiness_2026_09/moving_admission_risk_v1/result_interpretation.md).

### Previous Kinematic Representation Test

I tested whether explicit causal motion features improve the neural model's
intervention-risk estimates. A 64-field basis describes past velocity changes,
turns, neighbor approach and baseline rollouts. I compared it with an otherwise
identical continued-training control in three matched contexts. Both arms
start from the same weights and use the same loss, sampling and update budget.
The original 2% risk and support requirements are unchanged.

Six real neural fits completed 12,000 primary updates. The new features reduce
full fitting loss by 30.13% to 72.99%, but do not improve overall held-source
utility:

| Risk model | Known ADE gain over floor | Hard gain | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|---:|
| Previous affine readout | 4.32% | 5.35% | 22.01% | 0/17 |
| Equal-budget continued-training control | 3.96% | 5.13% | 18.51% | 0/17 |
| Causal kinematic representation | 3.94% | 4.86% | 17.32% | 0/17 |

**This model repair does not pass.** Only one of three fixed contexts meets
the prospective expansion condition. Known all-risk violations increase from
13 to 14; easy violations fall from 16 to 14, but two selected-easy views also
become undefined. Worst defined recording easy-net degradation is 37.35%, not
a safety pass. I retain all contexts rather than promote the favorable one.

The fitting-policy comparison also worsens despite the lower loss. On the
actions it admits, the model underestimates easy harm even in fitting data;
unseen-scene shift alone cannot explain the failure. More training without the
new features also loses held utility against the warm model. These findings
point to a mismatch between average risk regression and conditional admission
decisions, not evidence that causal motion is generally useless.

Checkpoint resume, a full independent refit and all inference/decision/metric
replays are exact; 17 focused tests pass. This is new risk-encoder training,
not new trajectory dynamics. The comparison covers seven already exposed
SOURCE units and one producer seed. It is development evidence, not an
independent test. Missing-outcome completion bounds are not confidence
intervals. Deployment is unchanged; no identity-fallback certificate is used.
[Results, losses and failure analysis](outputs/publication_readiness_2026_09/moving_kinematic_risk_v1/README.md).

### Previous Nonlinear Readout Test

I tested whether a nonlinear risk model can recover useful information from
the frozen neural representation. In three matched contexts, I fitted one
readout using the two existing risk scores and another using those scores plus
64 hidden features. Inputs, source splits, proposals, supervision and the
original 2% risk requirements stay fixed.

The hidden-feature readouts reduce fitting loss by 40.87% to 64.59%, but that
advantage does not transfer to the seven exposed held SOURCE units:

| Risk readout | Known ADE gain over floor | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|
| Previous affine readout | 4.32% | 22.01% | 0/17 |
| Nonlinear scores only | 1.58% | 8.06% | 0/17 |
| Nonlinear scores + hidden features | 1.51% | 8.46% | 0/17 |

**This repair fails.** Both new readouts discard many useful interventions
while still admitting actions that exceed the harm budget. Adding hidden
features is slightly worse than the matched score-only control on held-source
known utility, despite helping on the fitting data. Known all/easy risk
violations remain 13/15, and two selected-easy views become undefined. Fewer
interventions and lower average prediction loss are not safety certificates.

Six real readout fits, checkpoint resume, a full independent refit and exact
inference/readout replay are verified. These are standard boosting readouts,
not new encoder training or a new dynamics method. The comparisons use one
producer seed on exposed development data; they are not independent-test or
multi-seed evidence. I have not changed the rules or upgraded deployment.
[Results and mechanism analysis](outputs/publication_readiness_2026_09/moving_nonlinear_readout_v1/README.md).

### Previous Affine And Direct Risk Tests

An exact affine-output refit previously raised known floor gain from 4.09%
to 4.32%, but left full risk/support at 0/17 and failed its expansion criterion.
That control helps distinguish final-layer optimization from the unresolved
problem of predicting risk on the subset selected for intervention.
[Affine control](outputs/publication_readiness_2026_09/moving_readout_refit_v1/README.md).

I tested a more direct way to learn intervention risk: predict how far an
action lies above or below the original harm budget. Six neural heads compare
ordinary mean regression with a fixed upper-quantile objective, with matched
initialization, sampling and 2,000 updates per head. This is a deliberately
limited model-repair experiment on seven already exposed SOURCE units, not an
independent benchmark or a full multi-seed result.

| Policy on the same pilot sources | Known ADE gain over floor | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|
| Previous proposal-conditioned critic | 5.33% | 31.79% | 1/17 |
| Direct mean-risk head | 4.09% | 21.31% | 0/17 |
| Direct upper-quantile head | 0.15% | 1.52% | 2/17 |

Neither repair is deployable. The upper-quantile head discards most useful
interventions yet still violates the original risk requirements. The problem
also appears on its fitting data, so unseen-scene shift is not the only
explanation. Marginal prediction quality does not ensure accurate risk on the
subset a model chooses to admit. I retain the failed and undefined cases,
including the quantile policy's two unavailable matched utility comparisons.

All six fits, the independent full refit, checkpoint resume and readout replay
are verified. The original 2% rules remain unchanged; there is no identity-
fallback certificate or deployment upgrade. The pilot does not meet its
prespecified condition for expansion.
[Results and failure analysis](outputs/publication_readiness_2026_09/moving_direct_tail_risk_pilot_v1/README.md).

### Previous Full Comparison

I completed a controlled test of proposal-conditioned risk learning: the risk
model learns only the actions proposed by a router fitted on different source
scenes. I trained 108 neural critics for 216,000 updates, keeping the forecaster,
initialization, sampling, source splits and original 2% requirements fixed.
Training took about 128 minutes locally, with exact checkpoint and inference
reproduction.

The change did not improve the policy. Compared with the all-action critic,
the known-label utility difference is -0.29 percentage points (95% source CI:
-0.86 to +0.28). Only 28/696 repeated recording views meet the full original
risk/support requirements, down from 38/696. Easy-risk violations remain high,
and eight required source views still lack an estimable matched comparison.

The failure is informative but not a success claim: the critic overestimates
harm across the proposal pool, then underestimates it on the actions it admits.
Restricting its training actions did not resolve that problem. **The original
rules and previous failures are unchanged, and I am not upgrading deployment.**
These are exposed MovingDroneCrowd++ development results, not independent
confirmation or a new safe world-model contribution.
[Full results, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_proposal_risk_v1/README.md).

## Previous Controlled Tests

An earlier comparison tests a published cost-sensitive routing objective on the
same frozen MovingDroneCrowd++ forecast bank. I trained 54 source-isolated neural
routers, with 108,000 updates and three producer seeds. The original point
controls and all new inference blocks reproduce exactly.

Direct routing improves available-label ADE over the physical floor by 12.29%
(95% source-bootstrap CI: 6.94 to 17.88). Against the previous point selector,
the paired known-label gain is 5.25 percentage points [1.31, 9.55]. Those are
exposed development results, not independent confirmation. The direct
missing-outcome lower comparison against the point selector still crosses zero.

The safety result is less encouraging. With the same risk veto, the paired
advantage is only 0.82 points [-1.05, 2.93], and just 38/696 repeated recording
views meet the original complete risk/support contract. There are 483 known
easy-risk violations. Eight required source views have empty matched controls,
so that complete comparison remains unavailable. **I have not changed the 2%
rule, certified empty cases as safe, or upgraded deployment.**

This control shows that better mean routing utility does not automatically give
reliable admission decisions. It adapts an existing public method; it is not a
new M3W loss or a new trajectory-world-model contribution by itself.
[Full results, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_public_deferral_v1/README.md).

### Joint Harm Supervision

The preceding controlled experiment tests whether explicitly learning the chance
and size of harm makes neural interventions more reliable. I trained 108
matched cost heads on MovingDroneCrowd++ SOURCE, with the same initialization,
sampling, fitting sources and update budget in both arms. Event supervision
improves harm-estimation errors over the matched product-moment model, but
does not solve the decision problem. Its occurrence probabilities are also
worse than fitting-source frequency priors on the held development sources.

At the same intervention count and amplitude distribution within each query,
the new veto loses 1.03 percentage points of known-label utility against its
matched control (95% CI: -1.89 to -0.37). The complete nonempty primary remains
unavailable because 24 required source views have empty controls. Only 47/696
repeated recording views meet the original risk/support contract. I retain
those failures rather than changing the 2% limit or certifying empty cases as
safe. Deployment is unchanged.
[Complete experiment, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_joint_harm_v1/README.md).

This follows a [negative source-separated critic experiment](outputs/publication_readiness_2026_09/moving_source_separated_critics_v1/README.md).
The remaining problem is not simply average error prediction: risk estimates
can be conservative over the candidate pool yet optimistic on the actions
the model admits. These are exposed development results, not independent
confirmation or a new trajectory-world-model success.

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

### Moving-Camera Source Check

I have also completed the matched forecasting controls on 64 fit-admitted
MovingDroneCrowd++ recordings. Both model families use the same causal history,
neighbors and fitting budget: nine PCT and nine fixed-head EqMotion fits, with
three actual seeds and 72,000 primary updates. Every saved forecast replays
exactly, as do two complete first-model refits.

There are 195,286 past-eligible target windows, including partial and missing
future labels. Only 58 recordings in 23 source groups support the fixed 8/12
task; one declared group has no usable window. The original 24-group primary
is therefore not estimable. I report the supported-group results as secondary
development evidence, not an independent final test.

|Model|Available-label pixel ADE|Mean groupwise gain over causal CV|
|---|---:|---:|
|Causal CV|61.3680|0.00%|
|PCT|77.3815|-39.27%|
|Fixed-head EqMotion|53.4925|+15.74%|

EqMotion's known-label paired advantage over PCT is +55.02 pp
[+27.89, +87.39], but the missing-outcome lower contrast is +14.46 pp
[-8.19, +38.93]. That unresolved interval matters: this is a stronger proposal,
not a passed safety gate. Native annotation steps are not verified video time,
and the derived forecasting task is not the dataset author's published task.

[Full new-source comparison and loss curves](outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1/readout.md)
and [relative-motion diagnostic](outputs/publication_readiness_2026_09/moving_relational_diagnostic_v1/readout.md).
I also tested past-only shared-motion extrapolators, selecting their coefficients
on producer-fitting sources only. They do not beat CV on held sources: the
selected control's mean groupwise gain is -2.58%. EqMotion retains an observed
advantage, but its paired missing-outcome lower interval still crosses zero.
The [expanded baseline comparison](outputs/publication_readiness_2026_09/moving_social_causal_baselines_v1/readout.md)
keeps that limitation explicit rather than presenting a larger candidate list
as proof of a stronger baseline.
The subsequent matched cost study is also complete: 36 real Torch heads,
72,000 updates and exact full refit/replay checks. Adding 48 observed-motion
fields does not establish a gain: the known paired effect is -0.41 pp
[-2.66, +2.00], and the completion-lower effect is -5.33 pp [-8.95, -2.27].
Only 9/348 repeated recording views satisfy the shared-motion joint policy's
full original risk contract. It is not deployed.

The [failure diagnosis](outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1/failure_analysis.md)
finds severe underprediction specifically on policy-selected samples, already
within critic-fitting sources. Global cost fitting can look reasonable while
the apparently safest choices are wrong. This makes selection-conditional risk
learning the next problem to solve, rather than simply adding more motion cues.

An equal-budget follow-up tests stable logit supervision from the same 18
frozen cost heads. It improves fitting on the old selected cases, but the new
policy still underestimates harm on its own choices. Its switch rate falls
from about 70% to 12%, while known utility falls by 9.31 pp against ordinary
continuation. The paired missing-outcome lower effect is -16.86 pp
[-27.30,-7.69], and the complete risk contract still fails. I do not deploy it.
The [matched continuation report](outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1/failure_analysis.md)
show why improved cost fitting is not the same as safe intervention.

The next check exposes a more basic limit of the fixed binary proposal.
Even a label-aware choice can satisfy the original complete recording gate
in only 299/348 repeated views. In 15 views there is no observed easy event;
in another 34 every legal easy neural prediction exceeds the 2% harm ratio.
These involve 14 of 58 actual-query recordings, not 49 independent sites.
I keep all of them in the result rather than filtering to the feasible cases.
The [exact feasibility check](outputs/publication_readiness_2026_09/moving_record_feasibility_v1/readout.md)
shifts the next work toward candidate quality and event support, not another
blind selector-loss sweep. It is a diagnosis, not learned model performance.
Fixed past-only intermediate forecasts then expand that label-aware opportunity
to 330/348 views, covering 31 of the 34 candidate-limited cases. However, every
uniform amplitude still fails the complete contract. The
[amplitude controls](outputs/publication_readiness_2026_09/moving_amplitude_opportunity_v1/readout.md)
motivate learning the size of an intervention as well as its occurrence;
they are not a deployed selector or a safety guarantee.

I tested that idea with 36 fresh amplitude-conditioned cost heads and 72,000
updates. Both controls see the same paths and have the same capacity; one
learns/selects only the full neural action, while the other learns/selects
intermediate amplitudes. All new outputs and policy scores replay exactly.

The matched joint-policy known effect is only +0.46 pp [-0.48,+1.49]; its
paired missing-outcome lower effect is -1.16 pp [-2.91,+0.22]. With the same
amplitude-head weights, the richer action menu has a small observed benefit
of +0.59 pp [+0.30,+0.92], but its missing-outcome lower interval crosses zero.
Neither finding supports a new safe deployment: only 10/348 repeated recording
views meet the full original risk contract. The richer menu increases the
source-average switch rate from 63.24% to 70.93%, rather than simply shrinking
every intervention. I retain the failed primary and do not deploy this head.

[Complete amplitude-learning results and loss curves](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/failure_analysis.md)
and [reproduction guide](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/operation_zh.md).
The next question is why selected harm remains underpredicted on fitting
sources, not whether another held-threshold sweep can make this result pass.

The [frozen fitting-curve check](outputs/publication_readiness_2026_09/moving_amplitude_shape_v1/readout.md)
now finds that roughly 70% of the supervised amplitude head's repeated fitting
rows violate positive-harm convexity,although the true-label curves do not.
This is not a unique cause of failure. I tested a same-capacity constrained
decoder with 18 fresh heads and 36,000 updates, retaining the verified old
amplitude heads as matched controls. Curve checks pass, but known utility is
essentially unchanged: -0.02 pp [-0.53,+0.51]. The paired missing-outcome lower
effect is -1.83 pp [-3.62,-0.42], and complete risk support stays at 10/348
repeated recording views. I do not deploy it or claim new dynamics lift.

[Complete constrained-head results and loss curves](outputs/publication_readiness_2026_09/moving_convex_amplitude_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_convex_amplitude_v1/failure_analysis.md).
This separates a correct structural prior from reliable intervention costs:
enforcing the right curve shape is not enough to learn the right magnitudes.

I followed this with 54 small tree refiners that keep the forecasts and
reference costs frozen. The experiment compares global, fixed-region and
refreshed-region learning of intervention harm. A same-selection comparison
confirms that the new estimator corrects much of the extreme fitting error;
the change is not just avoiding difficult samples.

That does not yet make the controller reliable. Refreshed weighting gains
0.070 percentage points over fixed weighting on known labels, but the paired
completion-lower interval still crosses zero. Worst signed easy-net degradation
is 1.37%, while positive-harm and support requirements still fail. Only 47 of
348 repeated recording views meet the complete contract. All 72 held cost
outputs and policy metrics replay exactly. I retain this as a diagnostic
result, not a new neural world model or deployment upgrade.

[Complete experiment](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/failure_analysis.md)
and [common-selection checks](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/fitting_check/readout.md).

I tested whether past/current crowd information helps this cost
estimator. The new comparison retrains equal-width local-zero and shared-context
controls, leaving forecasts, targets, data roles and the 2% requirements fixed.
All 36 refiners and both complete refits are finished. The complete held
readout does not establish a feature gain: shared minus local is -0.042 pp
on known labels, and the direct completion-lower effect is -0.054 pp with a
95% interval crossing zero. Shared context improves worst signed easy-net
degradation to 0.90%, but the complete risk contract still fails. All 54 cost
outputs and 108 decision metric blocks replay exactly. This remains development
evidence, not a deployment upgrade.
[Protocol](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/protocol.md).
[Complete results](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/failure_analysis.md).

The follow-up fixed-decision decomposition identifies reference support as a
separate problem: 47 active views lack observed selected easy reference even
though every selected row has some future label. I tested a two-part reference
head against direct regression with source-excluded internal validation. The
direct head is better; the event-probability head does not establish a Brier
improvement. C-selected reference prediction beats a constant prior, but loses
to the existing learned neural reference on both known-label and conservative
bounded targets. The original neural reference remains the stronger component;
better-than-prior loss is not enough. Forecasts and policies stay fixed; this
is not a deployment upgrade.
[Risk-factor diagnosis](outputs/publication_readiness_2026_09/moving_record_risk_factors_v1/readout.md)
and [complete learned-reference comparison](outputs/publication_readiness_2026_09/moving_hurdle_reference_v1/strong_control/readout.md).

A fixed-selection follow-up finds that lower row-level reference loss can
coexist with worse recording-level mass estimates. I tested a targeted
change to the existing neural cost objective: learn aggregate excess on its
fitting teacher's selected sets, while leaving the network, observations,
forecasts and risk requirements unchanged. This is an experiment, not a new
deployment result. The fitting-only diagnosis then found that the model was
more optimistic on its own selections than on the teacher's selections. I
therefore trained one matched follow-up using the current detached selection
in that extra loss. Both banks are now frozen: 36 new neural cost heads across
three seeds, with the same forecasts, inputs and training budget. Their
held-source comparison was fixed in advance. The frozen-teacher loss improves known
utility against the conservative tree, but loses to the matched neural head,
does not establish the completion-lower gain and badly fails recording-level
easy preservation. The current-selection version is also complete. It recovers
known utility over that failed loss, but still loses to the original neural
head and does not repair risk. Its own fitting-region harm estimates are even
more optimistic. I do not deploy either version or interpret successful
training and replay as a successful method. The next test should separate
proposal selection from risk evaluation on genuinely source-excluded fitting
data, rather than assume another loss sweep will solve this.
Before retraining, I also tested a second frozen head as a risk veto and
matched its intervention count in every query. That cheaper alternative loses
utility even on C: it removes some harm but rejects more useful predictions.
Both heads share C training, so this is not a test of an independently fitted
critic. I do not scale or deploy this particular veto based on a lower harm
numerator.
[Same-count veto result](outputs/publication_readiness_2026_09/moving_cross_critic_veto_v1/readout.md).
[Complete two-study analysis](outputs/publication_readiness_2026_09/evidence_manuscript_v9/selected_record_objective_addendum.md)
and [current-selection results](outputs/publication_readiness_2026_09/moving_adaptive_record_cost_v1/readout.md).

A separate verified support check also limits what training can establish:
15 of the 348 recording views have zero full observed easy-reference mass.
No selected subset can create positive reference mass in those fixed views.
I retain them as unsupported rather than claiming that a smaller learning
loss makes the original complete-roster risk requirement pass.
[Support diagnosis](outputs/publication_readiness_2026_09/moving_support_ceiling_v1/readout.md).
[Selected-set evidence](outputs/publication_readiness_2026_09/moving_selected_reference_v1/readout.md)
and [matched neural objective](outputs/publication_readiness_2026_09/moving_record_excess_cost_v1/protocol.md).

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
- On the new moving-camera source, relative-motion prediction improves on
  observed labels, but its unknown-outcome lower contrast is -24.87 pp
  [-46.62, -5.83]. This does not establish a robust positive effect. The proxy
  is also blind to common translation and cannot replace absolute error or
  the full recording-level risk checks.
- A complete fitting-side audit finds that clipped inputs account for only
  0.95% of selected known easy-harm mass. Outliers exist, but this does not
  support another clipping-map training sweep as the main repair.

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
The completed European study uses four CPU compute threads, one inter-op thread and no
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
The earlier European scoped suite had 75 passing tests and one optional Metal-parity skip.
New-source checks and actual full replays are recorded with each experiment.
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

I am checking MovingDroneCrowd++ as a possible new development source rather
than reusing protected confirmation recordings. Its annotation intake contains
120 clips and 245,018 complete native 8/12 windows, not that many independent
scenes. All 360 sparse source views are now available; 21 portrait views needed
a fixed axis correction before matching. All 20,580 source-overlap pairs are
now complete, with full saved-result verification and a repeatable read-only
check. The 69-minute matching job hit a summary-assembly error after saving
every pair; a registered repair recovered the work without changing matching.

The screen finds two strong and 29 ambiguous pairs within the new source,
but no supported cross-source pair. This is not proof of independent sites.
The 64 publisher-train candidates retain 136,295 complete native 8/12 windows;
they are not independent samples. At the source-screen checkpoint they were
not yet admitted to training.
No new forecasting score or deployment result follows from this audit.
The DUT bridge is now complete: all 28 author backgrounds (19.4 MB), 6,874
fixed pairs and exact source/receipt replays. No supported cross-source link
or reservation conflict was found, with crop/composite/matching limits retained.
A background composite is not a past video frame and will not become a causal
model input.

The separate [fit-only admission](outputs/publication_readiness_2026_09/moving_development_admission_v1/readout.md)
now opens those 64 clips for local development fitting. Their 24 conservative
groups have a fixed proposed threefold allocation of 45,606 / 45,359 / 45,330
complete windows. Other clips remain unadmitted for prediction in this track;
DUT and DroneCrowd retain their original roles. The next experiment must bind
the actual predictor/teacher/critic exclusions before fitting. Source access
is ready; a new neural benefit or safe policy has not yet been demonstrated.
The [first admitted-source forecast comparison](outputs/publication_readiness_2026_09/moving_source_forecast_v1/protocol.md)
is now registered: matched PCT and fixed-head EqMotion, three source folds and
three actual seeds, 18 complete 4,000-update fits. It includes partial and
wholly missing future labels rather than keeping only complete trajectories.
Its matrices will stay in RAM; a real-data optimizer/resume pilot precedes
the full run. No new training result is claimed at registration.
Preparation now exposes one empty source group: its three clips have only
10-17 native frames, too short for the fixed 8/12 request. All 195,286 data
rows are intact. A forward support repair keeps the original 24-group contrast
not estimable and labels the 23 supported-group contrast secondary; it does
not turn the missing group into zero error or shorten the task.
The real native-arm64 Torch pilot now passes for both families: 50+50-step
resume and direct100 produce identical model/optimizer/sampler states and
outputs. Peak RSS is 1.46 GB; measured speed projects about 3.1 hours for all
18 full fits. The complete matrix is running: all nine PCT producers have now
finished their full 4,000 updates, and EqMotion continues. Pilot prefixes are
reused only for their identical first primary jobs. Pilot loss is fitting evidence, not a held
prediction or risk result.
[Complete result](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/readout.md),
[inspection scope](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/pair_review.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/tutorial_zh.md).

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
