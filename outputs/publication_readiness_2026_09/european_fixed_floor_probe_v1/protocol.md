# Fixed-Producer Floor: Incremental Opportunity and Learnability

Registered before new fitting/readout. This is a source-development diagnostic,
not independent confirmation. All twelve source-training localities have already
been opened; independent selection/calibration/confirmation roles remain closed.

## Why This Is Not the Previous Target-Reference Experiment

The September25 floor-relative study used older forecast banks and an inner
two-source floor versus an outer four-source floor. Its matched target repair
failed. This study uses the coordinate-unit-repaired nine forecast banks and,
critically, the SAME frozen floor checkpoint family to generate both new-probe
fitting and held targets. No in-sample floor outputs enter probe supervision.
It tests a small linear learnability probe, not a new neural world model.

## Source Roles

For each of nine producer/seed jobs and two legal controller assignments:
- Four localities fit the frozen neural forecaster.
- Four disjoint localities fit all frozen utility/easy heads; four fixed risk
  components each fit three of these controller sites.
- The remaining four localities are excluded from this entire producer chain.
- All six two-fitting/two-held rotations fit the new incremental probe.

The floor is CV or damping according to the frozen movement/utility/easy guard,
signed-excess ensemble score <=0 and controller-fitted99% support rule. It has
no probe-source calibration. The parent damping calibrated-supported action is
expected to equal this floor (all parent cutoffs were0); verify, do not assume.

## Prespecified Models and Actions

108 joint ridge fits produce216 matched five-output cost heads (CV vs floor
reference). Identical380 past-only features, equal fitting-locality weights,
training-only mean/std, feature clipping10, squared loss, regularization0.01,
and fitting-only mean CV error scale. Outputs: positive benefit, positive harm,
reference cost, CV-defined easy-event reference cost and easy-event harm.
Unknown future-label rows do not train and remain in decision denominators.
Use causal max rollout disagreement to bound benefit/harm; clamp costs >=0.

Positive policy: last-step movement and predicted benefit>harm. Safe policy adds
both predicted all/easy harm-to-reference <=2% and a99% fitting-input support
filter. Thresholds are fixed. No new threshold search or held selection.
Both policies fall back to the frozen damping floor, never silently to CV.

Controls: CV, frozen protected damping, parent neural-original CV fallback,
parent neural-rebased floor fallback, calibrated parent rebase, raw neural,
CV-target probes, floor-target probes, and future-label floor/neural oracle.
The oracle chooses by ADE and uses that same trajectory for FDE; not a model.

## Outcomes

Primary: floor-safe vs matched CV-safe paired all-ADE gain. Also compare every
action directly to the frozen floor. Decompose captured benefit, selected harm,
missed benefit, lost/gained default action and oracle opportunity. Report all,
easy, hard, complete/partial label sensitivity, endpoint FDE, intervention,
zero-reference harm, selected incremental harm ratio, tail/worst-source error.
Report regression MSE skill against fitting-only constants and gain ordering
AUROC; success on prediction moments is not a safety certificate.

Average dependent views within each of12 localities, then bootstrap3,000 paired
locality means, seed81531. Include all three forecaster seeds17/29/43; these are
not independent source replications. No winner/deployment is selected here.

Freeze registration before fitting, then all108 score/decision groups before
held scoring. Resume at completed group boundaries; heartbeat each group;
preserve10GiB disk. CPU4/interop1, workers0, arm64 native Python. Checkpoint
every completed closed-form fit; no gradient steps are claimed. Full exact
inference replay plus at least one fresh full fitting replay required.

Fresh_run: ridge fits, score inference, decomposition and bootstrap.
Cached_verified: raw-derived features, forecasters and floor producer chain.
Not_run: new trajectory training, independent validation, deployment, full
legacy suite and cold raw reconstruction. Silver image-local obs8/pred12 at
raw-frame stride12. No historicalt50, seconds, metric, human gold, physical
safety, true3D or foundation claim. Stage5C and SMC remain disabled.
