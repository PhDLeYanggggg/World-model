# Frozen-Policy Scene-Risk Screening

2026-09-16. `fresh_run` implementation, synthetic integration tests and analytical support calculation. Existing real recording arrays were `cached_verified` by metadata/content hashes. Real forecasting, policy calibration and independent confirmation remain `not_run` because the scientific protocol is unapproved. This is not a new predictive result or deployment approval.

## What Is Connected

The calibration entry consumes the actual development export, not an independently reconstructed policy. Before reading calibration labels it verifies the selected-policy hash, development report, frozen candidate plan, producer manifests, source-code identity and recursive fitting/selection exposure. Changed models, policy order, protocol, costs or development evidence are refused. Calibration cannot fit predictors, normalizers, costs or thresholds.

All past-supported agents participate in the scene decision before labels are read. The same queries and baseline must be shared by every frozen policy. Missing future labels do not remove agents from inference or from the risk denominator: an intervention with unknown error receives the worst bounded loss, one. A nonintervention has exactly zero excess over the unchanged baseline, including when absolute forecasting error is unknown.

The protocol must explicitly choose the primary loss, past-only normalization, query stride, recording-local graph geometry, bounded risk functionals, tolerances, within-scene aggregation and development-prioritized policy order. Supported risk functionals are clipped positive excess loss and a harm-event indicator with a declared margin. Neither is automatically a guarantee of at most 2% relative easy-case ADE/FDE degradation.

After approved within-scene aggregation, physical scenes receive equal weight. For M frozen learned policies, K risks and n independent calibration scenes with losses in [0,1], the implementation uses the simultaneous Hoeffding/union upper bound:

```text
U = min(1, mean_scene_loss + sqrt(log(M*K/delta) / (2*n)))
```

Select the first accepted policy in the order frozen by development, or use the unchanged baseline. An explicitly baseline-only policy has analytically zero excess loss; a learned policy with zero observed interventions does not inherit that exemption. A model may intervene on unseen scenes, so its sampling uncertainty remains.

This is a conservative bounded-loss screen, not a new theorem or a full implementation of Conformal Risk Control/Learn then Test. Independent, same-population scenes and a frozen policy family are assumptions, not facts established by assigning scene IDs. [Learn then Test](https://arxiv.org/abs/2110.01052) and [Conformal Risk Control](https://arxiv.org/abs/2208.02814) supply relevant statistical context; their guarantees cannot be borrowed without checking their hypotheses. Arbitrary cross-domain shift, physical safety and raw unbounded forecasting error are not certified here.

## Recovery and Provenance

The single-process CLI freezes code, ordered policies, artifacts, protocol and runtime in its run identity. It writes per-policy/recording row caches with hash receipts and PID/progress heartbeats. Interrupted runs reuse verified complete recordings and recompute unfinished recordings; completed runs verify the same result and return `cached_verified`. Recovery after the completed claim but before the completion marker does not repeat evaluation. Corrupted cache bytes are rejected.

The calibration report is registered as an artifact whose parents are the frozen development policies and whose calibration exposure is explicit. It can be checked for use on later confirmation data; that lineage check is not execution of confirmation. Output paths remain inside the workspace. These checks prevent accidental pipeline misuse; they are not operating-system access control against arbitrary code reading the filesystem.

## Verification Performed

- 21 new cases plus existing related regression checks: **157 passed in 37.97 s** on arm64 CPU. No new MPS calibration claim; earlier model/runtime MPS checks are unchanged evidence.
- The integration fixture runs actual synthetic forecaster training, two-fold out-of-fold cost construction, development selection, policy export and frozen calibration. The learned-policy case uses three 80-update forecasters and a constant-position floor to exercise a nonfloor policy; ordinary fixtures retain eight updates. These synthetic settings are not strong-baseline comparisons or scientific protocol choices.
- A fixture initially had empty easy support and correctly selected the floor. Its synthetic easy definition was changed before fitting to exercise the learned branch. No real threshold or real split was changed, and no synthetic accuracy is promoted to a project result.
- Tests verify decision-before-label ordering, label-missing worst-case loss, shared query support, scene grouping, rejection of changed provenance/implementation, no weight updates, small-sample refusal, resume and corrupt-cache rejection.
- Real preflight returns exit 2: `Explicit protocol approval required`. No real calibration or confirmation label was evaluated by this entry.
- The historical full-suite result remains 1,870 passed / 1 unrelated failure. It was not rerun or relabeled as all green; the old suite can rewrite experiment outputs.

See [machine-readable verification](verification.json), [support audit](support_audit.md), and [tests](../../../tests/test_m3w_risk_calibration.py).

## Support Finding and Research Consequence

The rebuilt current collection has nine canonical recordings but only six physical-scene groups, all historically development-exposed; zero scenes are assigned to approved calibration. Even the deliberately optimistic hypothetical of using all six as independent calibration scenes with zero observed loss gives a bound of 0.4996 for one policy/one risk at illustrative delta=0.05. More overlapping windows or bootstrap replicates do not increase n.

A synthetic dependence diagnostic copies each of six Bernoulli scene outcomes into 1,000 identical windows. Across 20,000 simulations, incorrectly treating those copies as independent accepts a risk-0.20 policy under tolerance 0.10 in 26.675% of runs; correct scene-level screening accepts none in this construction. This is an illustration of pseudo-replication, not a real M3W failure-rate estimate.

The sample-count table concerns this conservative bound at zero observed loss, not an impossibility theorem for every risk method. It nevertheless prevents claiming useful finite-sample safety from the present evidence. The shortest research path is to agree the scientific task and roles, test empirical baseline-relative gains with properly qualified scene-level uncertainty, and obtain independently reviewed scenes before stronger confirmation/calibration claims. Whether formal risk guarantees remain a main contribution is awaiting the user's decision; no scope change has been silently approved.

Stage5C and SMC remain off. Dataset-local/raw-frame limitations remain; no metric, seconds-level, true-3D, foundation or submission-readiness claim is made.
