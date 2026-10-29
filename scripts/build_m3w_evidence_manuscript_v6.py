"""Build a non-submission-ready research manuscript from verified frozen evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_m3w_cost_decision_evidence import build, sha, PUBLIC, STUDIES

DEST = PUBLIC/'evidence_manuscript_v6'


def manuscript(evidence):
    held = {(r['study'], r['policy']):r for r in evidence['policies'] if r['scope'] == 'inner_held'}
    descriptions = ['Continuous moment cost', 'Exact-record decision',
                    'Published deferral loss adapter', 'Positive-support unit repair']
    table = ['| Fixed controller | Physical-only gain % | Mixed gain % | Mixed supported views |',
             '|---|---:|---:|---:|']
    for name, description in zip(STUDIES, descriptions):
        physical, mixed = held[name, 'physical_only'], held[name, 'physical_neural']
        table.append(f"| {description} | {physical['known_ADE_gain_floor_percent']:.6f} | "
            f"{mixed['known_ADE_gain_floor_percent']:.6f} | "
            f"{mixed['supported_record_head_views']}/{mixed['required_record_head_views']} |")
    successful = any(r['full_record_roster_supported'] for r in held.values())
    result_boundary = ('At least one policy supports its complete exposed internal roster; this is not '
        'an independent safety certificate.' if successful else
        'No reported full policy supports the complete required exposed internal recording roster. '
        'The study therefore does not establish a deployable safe learned upgrade.')
    return f'''# Learning When to Replace a Motion Baseline under Partial Future Labels

Anonymous research draft. This is a complete prose working revision, not a
formatted eight-page submission, an independently confirmed method, or a claim
that the M3W world-model objective has been achieved.

## Abstract

Mean trajectory accuracy does not determine when a learned forecast should
replace a competent motion baseline. A controller may improve aggregate error
while harming easy agents or concentrating risk in a particular recording.
Missing future labels further separate observed utility from a defensible
deployment claim. We investigate forecast substitution with a frozen causal
floor, six analytic motion candidates and one learned forecast. We formulate
baseline-relative benefit, positive harm and easy-reference moments; derive
elementary disagreement completion bounds; and compare continuous cost
learning, a published regression-deferral surrogate and exact-record
primal-dual decision training. A single-factor conditioning experiment changes
the numerical unit of a positive-support constraint without changing its
feasible set. Each comparison retains three controller seeds, six producer
contexts and a complete preregistered readout. The results expose a gap between
an available action bank, learned mean utility and complete selected-risk
support. {result_boundary} All reported intervals remain conditional SOURCE
development evidence. Independent calibration, confirmation and a successful
scene-joint mechanism remain necessary before a main deployment claim.

## 1. Introduction

Trajectory forecasting often evaluates a predictor through aggregate position
error. In deployment, however, an existing causal predictor may already be
accurate for many agents. Replacing all of its outputs can discard that
competence. The decision problem is not simply whether a neural predictor has
lower mean error, but where a forecast substitution is useful and what evidence
supports its risk on the selected population.

Three distinctions motivate this investigation. First, positive and negative
error changes need not cancel for the purpose of protecting easy examples.
Second, an average over scenes can hide a failing recording. Third, future
labels may be partial or absent, even though a causal policy must still make
its decision without consulting their availability. These issues are present
even when the trajectory predictor itself is held fixed.

We study a controlled multi-action setting. The available forecasts are fixed
before controller training, exact aliases of the floor cannot manufacture
interventions, and every controller receives only causal information. This
isolates cost estimation and decision training from a simultaneous search for
a more favorable trajectory producer. A label-aware oracle asks whether the
bank contains feasible useful choices; it is never an inference-time model.

The research question is whether a learned decision can capture useful neural
substitutions while satisfying a fixed baseline-relative positive-harm budget
on every required recording. The contribution under investigation is not a
combination of JEPA and Transformer modules. This revision instead provides
an explicit decision formulation, controlled real Torch comparisons, a
measured optimizer-conditioning diagnosis and reproducible negative evidence.
The broader agent-scene world-model goal remains unfinished.

## 2. Related Work

Graph-structured and attention-based trajectory models already incorporate
agent interactions and temporal context. Trajectron++ [1] includes heterogeneous
agents, dynamics and environmental information. AgentFormer [2] uses agent-aware
attention to model social and temporal dependencies jointly. These precedents
mean that adding neighbors or a Transformer is not sufficient methodological
novelty. This round does not reproduce either forecasting implementation, and
their published scores are not numerically comparable to our native protocol.

Regression with multiple experts is also established. Mao et al. [3] provide
single-stage and two-stage regression-deferral formulations and surrogate
analyses. We adapt their two-stage logistic loss as a matched published-loss
control, not a proposed loss or reproduction of their original experiments.
Cortes et al. [4] study expert imbalance through cost-sensitive deferral. MILD
is not implemented here; an imbalance-only claim would need that comparison.

Risk control requires care about the loss and its assumptions. Conformal Risk
Control [5] is not an automatic certificate for arbitrary selected positive-harm
ratios. Removing a low-harm row can shrink the reference denominator enough to
increase the ratio. Thus a monotone threshold argument cannot simply be assumed.
We report empirical full-policy recording checks without a population guarantee.

## 3. Problem and Causal Information Boundary

Let x contain the current and past agent history, legal neighbor context and
diagnostics of forecasts computed from that history. A frozen floor b(x) and
K candidate forecasts p_k(x) are available. The controller chooses a(x) in
{{0,...,K}}, with a=0 retaining b. Intervention here means changing a forecast
shown to a downstream user, not controlling the real agent.

The observation protocol requests eight past and twelve future steps on the
current SOURCE grid, with twelve raw frames between steps. Historical raw-frame
t+50 is a different supplementary task. Coordinates remain image-space or
dataset-local unless separately verified; this paper makes no meter, effective
second, true-3D or foundation-model claim.

For a nonempty observed future mask M, let c(p,y;M) be native partial-mask ADE.
An absent future outcome is not a zero-error label. Ground-truth future
endpoints, future goals, eventual label masks and central-difference velocities
are not controller inputs. Train-only fitted normalizers and legal causal
rollout endpoints are allowed. This estimand is not fully observed twelve-step
trajectory error, physical collision probability or real-world control safety.

For candidate k define B_k=max(c(b)-c(p_k),0), H_k=max(c(p_k)-c(b),0),
R=c(b), ER=E*R and EH_k=E*H_k. The easy event E uses a positive CV error
below a threshold fixed from producer-fitting data. Future labels define this
supervised event for loss/evaluation only; they never identify easy examples
to the deployed decision.

For a selected set, the positive-harm ratios are sum(H)/sum(R) and
sum(EH)/sum(ER). Net easy ADE change is different: gains on one easy row must
not erase positive harm on another for this ratio. The approved tolerance is
2% on every required recording. A zero denominator is not_estimable, not a
safe zero. Count-matched controls compare utility only, not safety.

## 4. Completion Bounds and Support

For any future point, the reverse triangle inequality bounds the absolute
change in position error by ||p_t-b_t||. Therefore any nonempty masked ADE
change is bounded by D_k=max_t ||p_kt-b_t||. The implementation additionally
retains the preceding CV disagreement envelope when larger.

For an unknown selected outcome, -D_k is a lower utility contribution and D_k
an upper positive-harm contribution. Combining known positive reference mass
with unknown upper harm gives a conservative ratio bound whenever the known
denominator is positive. This is a deterministic completion statement for
fixed decisions, not missing-at-random identification, a population certificate
or a claim that missing outcomes actually attain the bound.

A necessary support fact is equally important. If every genuine eligible known
easy action in a recording has harm/reference ratio above 2%, any nonempty
mixture remains above 2%; its ratio is a reference-weighted average. If no such
reference action exists, the selected easy ratio is undefined. These elementary
facts explain why an attractive average oracle and a feasible every-record
oracle are different diagnostics. They are not claimed as new consistency or
generalization theorems.

## 5. Controllers and Controlled Conditioning

### 5.1 Continuous Conditional Costs

A shared MLP predicts five nonnegative moments B,H,R,ER,EH from each action's
380-dimensional causal geometry/rollout feature vector and action identity.
Causal disagreement bounds its predicted error-change magnitude. Moment MSE
uses a fixed higher weight for underestimated harm and a within-row continuous
cost contrast. It does not classify a one-hot best-baseline label. Physical-only,
physical+neural and neural-only arms retain matched dimensions, initialization,
source-balanced row draws and 2,000 updates. All preprocessing is fitted using
physical fitting actions only.

### 5.2 Published Regression-Deferral Control

With the floor treated as expert zero, the logistic surrogate of [3], Section4,
Eq.(3), has coefficient sum_l c_l-c_j for expert j. We minimize the weighted
negative base-two log-softmax using known continuous ADE costs. Only the active
bank enters its loss. Unknown outcomes are not sampled as labels and remain
in evaluation. A separate preregistered fixed cost gate uses the earlier frozen
moment predictions; no held-result threshold search is performed. The native
loss application does not verify the original bounded-loss theorem assumptions.

### 5.3 Exact-Record Decision Training

The decision head freezes both producer and moment model. It receives twelve
causal descriptors per action: five predicted moments, six existing motion/
neighbor descriptors and a disagreement envelope, plus action identity. A
673-parameter MLP corrects a warm-start action logit. The floor logit is zero;
unavailable actions cannot be chosen. A straight-through categorical operation
uses hard actions forward and a temperature-annealed gradient surrogate.

For each fitting recording we accumulate complete finite-population lower
utility and all/easy harm-minus-budget totals, including unknown completion
terms. Separate recording duals enforce the two harm constraints. A third
constraint requires genuine positive selected easy-reference support, equal
to one minimum eligible easy-row equivalent. Full-gradient chunking changes
memory use, not the fitting population. It does not make the hard constrained
problem convex or its solution globally feasible.

### 5.4 A Single-Factor Numerical Repair

Write positive support as S_r>=m_r>0. The original normalized excess is
(m_r-S_r)/s_r, where s_r is the recording easy-reference scale. The repair uses
(m_r-S_r)/m_r. Both have the same feasible set. With a zero-initialized dual,
nonnegative support and step size eta, the old dual after T steps is at most
min(L,T*eta*m_r/s_r). The frozen fitting median empty excess is0.000292227;
372/444 repeated views per bank cannot reach a dual of1within500updates.
This diagnoses weak numerical pressure, not a sufficient cause of failure.

The repair changes only the third constraint's unit and corresponding effective
dual coefficient. Model, inputs, populations, all/easy2% budgets, optimizer,
dual cap and500updates remain fixed. An unchanged-unit500-update control exactly
reproduces the original scientific checkpoint state. All three real recovery
pilots match interrupted and continuous training. No checkpoint is selected
after looking at a favorable readout.

## 6. Experimental Protocol

These are SOURCE-development experiments, not a fresh independent benchmark.
Twenty-four hash-verified TRAIN packets supply twelve localities. Six frozen
producer/controller contexts and three controller seeds yield54fits per study.
The trajectory producer is frozen at seed43; three head seeds are not three
newly trained forecasters. Fitting and inner-held record roles are explicit,
but the latter have been exposed in earlier development and cannot be renamed
as independent confirmation or whole-pipeline out-of-fold evidence.

All54endpoints are frozen before the first readout of each new study. Every
completed144-view readout is replayed exactly. The main utility table averages
context/head views within locality before a3,000-resample locality bootstrap.
There are74unique fitting and31unique inner-held source/recordings. Repeated
contexts, head seeds and overlapping windows are not independent recordings.
Intervals are conditional development uncertainty, not a guarantee about unseen
domains, hidden labels or the forecaster population.

The banks hold candidate forecasts fixed; exact aliases of the floor are not
credited as interventions. We retain every preregistered seed and bank, fixed
sensitivity controls and missing-outcome completion contrasts. Physical-only
is the key comparator for whether adding the frozen neural expert helps.
Whole-policy and count-matched utility are reported separately.

## 7. Results

The following table is generated from the completed verified aggregates.
Gain is known-label ADE improvement over the floor, not a completion lower
bound. Supported views require both all/easy2%risk support, including the
prescribed missing-outcome treatment. The186views repeat contexts/head seeds.

{chr(10).join(table)}

Full per-bank intervals, fixed gated controls, completion bounds, intervention
counts, undefined risks and fitting monitors are in the accompanying evidence
table. {result_boundary}

The experiments must not be reduced to a ranking by mean ADE. The earlier
record decision collapses toward empty intervention; the published unrestricted
loss instead selects extensively and violates required risk slices. The
conditioning control tests a measured numerical weakness without changing the
evaluation contract. Its result does not by itself establish that all future
failures have that mechanism. Different controller families also have different
capacities and objectives; only the support-unit pair is a single-factor trial.

## 8. Failure Analysis and Limitations

Available forecasts are not the same as causally identifiable good choices.
A label-aware oracle uses outcomes unavailable at inference. Conditional
moment regression can underestimate harm on the population selected by its
own errors. Finite hard decisions and temperature saturation can make a
constrained gradient procedure favor the floor or oscillate between constraints.
Changing one numerical unit can repair pressure without repairing predictive
features, model capacity, missing-outcome uncertainty or record shift.

The studies do not yet implement scene-joint interaction optimization. Neighbor
features are not a joint multi-agent decision method. No new JEPA contribution,
Transformer dynamics lift, occupancy improvement, verified physical validity,
cross-dataset success or foundation-model capability is established here.
The native partial-mask estimand and conservative unknown completion can differ
substantially from a fully observed forecasting benchmark. Detector-derived
or visual-prior annotations are not human gold.

Independent model selection, risk calibration and final confirmation remain
unopened. A matched public forecasting implementation, explicit intervention
rate/conflict comparisons, independent-domain evidence and a successful causal
mechanism are still missing. This revision is useful research material but
does not satisfy the project's submission-readiness or world-model success gates.

## 9. Reproducibility and Responsible Reporting

Each real run records native arm64Torch CPU threads, single-process loading,
configuration, PID, heartbeat, atomic recoverable checkpoints, optimizer state
and data/feature/weight hashes. Published aggregate bundles preserve numerical
JSON bytes; a bundle check is not fresh inference. Full training or inference
needs the licensed/private local source assets and frozen checkpoints. The
public aggregate package alone does not provide end-to-end reconstruction.

The protocol retains historical failed exits and the weight-preserving singleton
inference fix. It does not rerun unchanged validated experiments merely to
change a provenance label. Stage5C execution and SMC remain disabled. Formal
submission requires a separate user confirmation; no paper has been submitted.

## LLM Usage Statement

LLM assistance was used substantially for code development and debugging,
experiment orchestration, literature retrieval, analysis, documentation and
draft preparation. This was not limited to grammar correction. Claims must
remain traceable to the frozen artifacts and primary sources; the statement
does not certify that all analysis is correct or that scientific review is complete.

## References

1. Salzmann etal. Trajectron++: Dynamically-Feasible Trajectory Forecasting With
Heterogeneous Data. ECCV2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf
2. Yuan etal. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent
Forecasting. ICCV2021. https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html
3. Mao,Mohri andZhong. Regression with Multi-Expert Deferral. ICML2024.
https://proceedings.mlr.press/v235/mao24d.html
4. Cortes etal. Optimized Deferral for Imbalanced Settings. ICML2026.
https://proceedings.mlr.press/v306/cortes26b.html
5. Conformal Risk Control. ICLR2024.
https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    evidence = build()
    export = PUBLIC/'cost_decision_evidence_20261029.json'
    if json.loads(export.read_text()) != evidence:
        raise ValueError('First build the exact frozen comparison export')
    value = manuscript(evidence)
    outputs = {'manuscript.md':value,
        'evidence.json':json.dumps(dict(aggregate_export_sha256=sha(export),
            result_source='cached_verified_manuscript_export_not_fresh_experiment',
            manuscript_sha256=__import__('hashlib').sha256(value.encode()).hexdigest(),
            independent_confirmation=False, submission_ready=False,
            author_details_in_anonymous_draft=False), indent=2)+'\n'}
    DEST.mkdir(exist_ok=True)
    for name, content in outputs.items():
        path = DEST/name
        if args.check:
            if path.read_text() != content:
                raise ValueError('Exact manuscript export differs')
        elif path.exists() and path.read_text() != content:
            raise ValueError('Preserve frozen paper revisions; use a new revision')
        elif not path.exists():
            path.write_text(content)
    print(json.dumps(dict(exact_check=args.check, words=len(value.split()), submission_ready=False)))


if __name__ == '__main__':
    main()
