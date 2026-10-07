# Technical Supplement: Costs, Selection, Missing Outcomes and Reproduction

This supplement describes the completed temporal-auxiliary comparison in the
October 7 development manuscript. It does not change that experiment, its failed
advancement screen or the active successor registration. The executable examples
are synthetic mathematical checks, not additional real-data evidence. Code paths
are relative to the research checkout; this is not yet an anonymous release.

## S1. Unit of Analysis and Targets

A row is an agent query with eight past annotation steps and twelve predicted
steps at raw stride 12. A query group is the current recording/frame, including
its supervised agents. TRAIN and recording-held development use disjoint whole
recordings inside already explored localities. Three cost-head seeds do not
constitute three independently trained trajectory forecasters. Window counts,
source/head occurrences and locality counts must not be conflated.

Let O_i be the nonempty set of observed future steps for row i. Reference and
candidate predictions r_it and n_it are fixed before the intervention head is
fitted. Their observed-support errors are

```
R_i = mean_{t in O_i} ||r_it - y_it||_2
N_i = mean_{t in O_i} ||n_it - y_it||_2
B_i = max(R_i - N_i, 0)
H_i = max(N_i - R_i, 0).
```

The label vector is Y_i = (B_i, H_i, R_i, e_i R_i, e_i H_i), with the fixed
TRAIN-derived easy event e_i. Wholly unknown futures are represented by an entire
missing label vector, never by zero cost. Partial observed-support ADE is not
complete-future ADE. Known partial-label costs remain fixed in the completion
calculations below; those calculations do not invent missing time-step positions.

Let D_i = max_t ||n_it-r_it||_2. The reverse triangle inequality gives
|R_i-N_i| <= D_i for any common nonempty observed mask, hence B_i+H_i <= D_i.
D_i uses two forecasts, not a future observation. No future mask determines an
inference action. Image-local coordinates and raw annotation frames do not
establish metre units, effective seconds, or physical safety.

## S2. TRAIN Preprocessing and Sampling

Preprocessing has one TRAIN locality per fitted source head. Known rows receive
equal query weights and equal within-query agent weights. With Q known query
groups and m_q known agents in group q, w_i = 1/(Q m_q). Unknown labels receive
zero fitting weight, but their causal inputs remain eligible for later decisions.
Feature means and standard deviations use these TRAIN weights, with standard
deviations floored at 1e-6. Feature z-scores are clipped to [-8,8].

The cost scale s is the TRAIN-weighted mean reference error; it must be positive.
All five targets and the disagreement envelope are divided by s. Eight TRAIN RMS
values normalize the five moments and three signed scores, each floored at 0.01.
The feature-support limit is TRAIN-derived. Neither validation statistics nor
future availability changes this preprocessing.

Each update samples 16 query groups with replacement and includes their known
agents. The objective first averages agents inside a sampled query, then queries.
This differs from a plain average over concatenated agents when query sizes vary.
The sampler uses seed+7919; a separate 128-query TRAIN monitor uses seed+9137.
Monitoring is not validation selection. The complete grid has 24 source contexts,
three seeds (17,29,43) and three auxiliary arms: 216 fixed-final fits.

## S3. Bounded Decoder and Exact Objective

The shared encoder is Linear(380,32) followed by SiLU. The primary output has five
logits a_0,...,a_4. Its weights start at zero and its bias is computed from TRAIN
means. Let (f_B,f_H,f_0) = softmax(a_0,a_1,0). In normalized cost units,

```
B_hat  = (D/s) f_B
H_hat  = (D/s) f_H
R_hat  = softplus(a_2)
ER_hat = R_hat sigmoid(a_3)
EH_hat = H_hat sigmoid(a_4).
```

Thus B_hat+H_hat <= D/s, ER_hat <= R_hat and EH_hat <= H_hat. These are algebraic
constraints, not calibrated conditional-risk bounds. The signed-score vector is
S(Y) = (B-H, H-0.02R, EH-0.02ER). For channel j, let d_ij be prediction minus
target divided by its frozen TRAIN RMS. The implemented primary loss is

```
L_primary = mean_q mean_{i in q} [
    (1/10) sum_{j=1..5} d_ij^2 + (1/6) sum_{j=6..8} d_ij^2 ].
```

It is not the unweighted mean of eight channels: moments and signed scores each
receive half of the total weight. The distinction matters when discussing a
single-channel loss replacement or interpreting a total-loss decrease.

The auxiliary linear output has 24 entries, reshaped to 12 steps by 2 channels,
and starts at zero. Its channels predict candidate-minus-reference step error
and reference step error. Temporal supervision uses those observed step targets;
row-mean supervision repeats each row's observed mean at its observed steps.
The no-auxiliary arm retains the same graph but multiplies its auxiliary loss by
zero. Auxiliary squared error averages its two channels and observed steps per
agent, then agents per query and queries per batch. Missing steps are masked
before arithmetic, not after propagating NaNs. Total loss is L_primary plus
0.1 L_auxiliary for row-mean and temporal, or plus 0 L_auxiliary for the control.

All arms use AdamW(lr=0.001, weight_decay=0.0001), gradient-norm clipping at 5,
2,000 updates, query batch 16, checkpoint/heartbeat interval 100. Primary initial
states, inputs, preprocessing and sampled query/row sequences match across arms.
No early stopping, best-validation checkpoint or post-readout threshold search is
part of this comparison. Four compute threads, one interop thread and zero loader
workers are execution settings, not changes to the scientific objective.

## S4. Actions, Same-Count Controls and Missing Outcomes

For causal motion-eligible, TRAIN-supported rows, the raw action is

```
switch_i = (B_hat-H_hat > 0)
           and (H_hat-0.02R_hat <= 0)
           and (EH_hat-0.02ER_hat <= 0).
```

The actual easy event is not an input. These predictions are not sufficient for
a safety certificate. Pairwise matched comparisons retain the smaller action
count within each recording/frame, selecting each arm's eligible rows by predicted
net benefit and breaking ties by stable row ID. This matches intervention count,
not selected identities, oracle gains or a global action rate.

For fixed action a, let U(a) = sum_{unknown i} a_i D_i. All sums below are row sums
within the evaluated source/head view, not the query weights used in fitting.

```
utility_lower(a) = sum_{known i} a_i (B_i-H_i) - U(a)
all_risk_upper(a) = [sum_known a_i H_i + U(a)] / sum_known a_i R_i
easy_risk_upper(a) = [sum_known a_i e_i H_i + U(a)] / sum_known a_i e_i R_i.
```

Unknown selected harm is conservatively bounded by D_i; its nonnegative reference
cost is not inserted into the denominator. Missing/zero selected denominators are
undefined, not safe. Full-population easy net degradation uses known easy harm
minus known easy benefit plus U(a), divided by all known easy-reference cost.
If any unknown rows exist, this upper numerator is clamped below at zero because
unknown unselected rows may enlarge the denominator. This net quantity differs
from positive selected easy risk and does not replace it.

For two actions a and b applied to the same forecasts/outcomes, their paired
utility difference has the finite-completion interval

```
C = sum_known (b_i-a_i)(B_i-H_i)
M = sum_unknown |b_i-a_i| D_i
paired_difference in [C-M, C+M].
```

Shared unknown selections cancel. With unknown envelopes (2,3,4), a=(1,1,0) and
b=(1,0,1), the paired interval is [-7,7], while subtraction of the two individual
lower bounds gives -1. That subtraction is not a lower bound on the paired
difference. The script enumerates the relaxed row-envelope extremes to verify
this example. It does not assert that every extreme is attainable under extra
scene geometry not encoded by those envelopes.

An empty intervention set cannot establish positive improvement. The screen
requires defined selected denominators, positive completion-lower utility,
all/easy positive-risk upper bounds <=2%, and easy degradation upper <=2%.
These are finite observed-cohort checks, not population-risk guarantees.

## S5. Why Threshold Tightening Is Not a General Risk Proof

Even a nested threshold family need not yield a monotone selected harm ratio.
Consider a fixed ranking with scores (3,2,1), easy harm (1,0,50) and easy reference
cost (10,1000,10). Thresholds (2.5,1.5,0.5) select successively larger sets, with
risk 10%, approximately 0.09901%, and 5%. The middle policy passes 2% and both ends
fail. This example concerns a ratio property; it is not an estimate of any M3W
policy or proof that calibration is impossible. It explains why a monotone-loss
calibration theorem cannot simply be assumed for this selected ratio.

Likewise, for step errors d_t = candidate_error_t-reference_error_t,
G_H = mean(max(d_t,0)), G_B = mean(max(-d_t,0)), and
H = G_H-min(G_H,G_B). Alternating errors (2,-1) yield G_H=1 but H=0.5. Penalizing
G_H as if it were the original H changes the scientific target, rather than
merely giving it more temporal detail. The completed auxiliary experiment kept
the original primary H and added separate temporal supervision.

## S6. Aggregation, Statistics and Negative Results

There are 72 source/head views across twelve development-exposed localities.
For each contrast, first average repeated views/seeds within locality, then take
the equal-locality mean. A paired bootstrap resamples twelve locality indices
with replacement, 3,000 times, using seed 20261005; the interval uses the 2.5th and
97.5th percentiles. Do not bootstrap overlapping windows as independent samples.
If any required source/head contrast is undefined, its full contrast is undefined;
unsupported heads are not silently removed.

The temporal screen required all stated prediction and utility contrasts against
all six controls, plus absolute full/matched risk checks. It failed. In particular,
better all-row MSE versus row-mean coexisted with worse full and matched utility.
The paper reports the entire comparator set, not only the favourable forest
comparison. These nominal intervals are development-exposed and not adjusted for
the preceding research search; they do not reopen independent confirmation.

The later 216-head TRAIN replay uses no optimizer updates. Its signed identity is
H_E-qR_E = (Hhat_E-qRhat_E)+(H_E-Hhat_E)+q(Rhat_E-R_E), q=0.02.
It diagnoses selected fitting error. It is neither held-out generalization nor
a causal allocation of blame among loss, features and optimization.

## S7. Active Successor: Motivation, Not a Result

The registered successor replaces only the fifth moment quadratic with
D(u,v)=2(u-v+v log(v/u)), using u=EH_hat/RMS_EH and v=EH/RMS_EH after cost
normalization. Zero labels have the continuous limiting term; wholly unknown
labels are not zero labels. Zero disagreement has identically zero harm.

Writing l=log(u), the derivative is 2(u-v). For squared error it is 2u(u-v).
When a positive target is severely underpredicted, the latter can have a very
small log-coordinate gradient. At l=-20,v=1, these are approximately -2 and
-4.12e-9. This is only a loss-geometry motivation. Network-chain gradients,
shared objectives, stochastic sampling and finite optimization remain relevant;
the example does not prove why the real heads failed or that the repair works.
The continuous-cost deviance is not asserted to be a Poisson count likelihood.

The successor requires 144 final fits before readout. Its explicit execution
amendment distinguishes 54 historical-exact controls from 18 exact original-trainer
replays after historical bitwise differences. Preserve that limitation; do not
call all 72 historical matches. Its new risk and utility result remains pending.

## S8. Reproduction and Release Boundary

From a checkout with the existing native environment:

```bash
.venv-pytorch/bin/python -m scripts.build_m3w_evidence_manuscript_v4 --check
.venv-pytorch/bin/python -m scripts.verify_m3w_supplement_v4 --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_supplement_v4.py
```

The first command reassembles cached aggregate evidence and verifies its sources.
The second runs synthetic identities and checks the linked source hashes. Neither
trains a model, reads real trajectories, opens independent labels, or reproduces
all remote weights. Exact numerical training replay additionally requires the
frozen licensed source data, role manifests, TRAIN packets, environment and saved
checkpoints listed in the experiment registries. See `reproduction_zh.md` for
owned CREATE status/collection and the frozen-readout sequence. A timeout is not
a terminal job state; inspect the same job before any recovery.

The public checkout includes source, configurations, aggregate receipts and paper
material, not raw data, image/video assets, private caches or model checkpoints.
An anonymous, portable release still requires removal of identity-bearing paths
and a clean-environment reproduction. Do not describe this supplement or these
synthetic checks as completion of that release. Stage5C and SMC remain off.
