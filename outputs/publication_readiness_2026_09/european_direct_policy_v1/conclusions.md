# Direct Neural Policy: Completed, Not Deployable

## Result

The registered primary method fails. Direct neural intervention training reduces
some error-related harm but rejects too much benefit, and its training safety
does not transfer to held recordings. Deployment is unchanged. This is not a
CCF-A submission-ready result or evidence for a new world dynamics model.

54 real neural policy heads,108,000 optimizer updates,three head seeds and three
fixed arms completed. Every final model was frozen before the144-view readout.
All144 groups and the summary replay exactly.88 relevant tests pass. These are
engineering and reproducibility checks, not a scientific pass.

## What Improved, and What Did Not

| Primary hard_record | Inner fit | Existing inner held |
|---|---:|---:|
| Full-policy budget views | 35/72 | 4/72 |
| Budget recording-views | 125/444 | 17/186 |
| Selected occurrences | 46,508 | 19,915 |
| Known all-risk violations | 28 | 43 |
| Known easy-risk violations | 5 | 50 |
| Undefined selected easy ratios | 9 | 11 |

The held full-policy utility difference versus original is bounded by
[-5.342713,-0.326202]pp. At equal per-record/frame intervention counts it is
[-1.132721,-0.625042]pp. Upper bounds are negative for all three head seeds.
These are partial-label completion bounds, not confidence intervals, statistical
significance, relative FDE gains or proof of future population behavior.

The controlled hard-versus-soft-record contrast improves matched held utility
by[0.012406,0.030458]pp. But its full difference spans zero and its known easy
violations increase39->50. Source-to-record constraints yield a small fit benefit;
held full and matched intervals span zero. Neither secondary result rescues the
primary failure or licenses replacing the preregistered method.

## Diagnosis

The primary retains20.90% of the original policy's normalized known benefit and
20.05% of its known harm on held recordings. Interventions fall32,621->19,915,
but conditional choice quality is also worse at matched counts. This is not
only a coverage reduction or a missing-label artifact:43 all-risk and50 easy-risk
violations occur on known labels.27 full easy-degradation upper bounds exceed2%.

Empirical optimization is not solved even on fit:28 known all-risk violations
remain. Stochastic dual constraints and hard forward actions do not guarantee
whole-recording feasibility. Attribution to minibatch noise, the dual cap or
feature insufficiency remains a hypothesis, not an established single cause.

## Evidence Scope

There are12 source localities,74 fit and31 held source-qualified recordings.
897,060 fit and419,766 held row occurrences repeat across model contexts/seeds.
The original TRAIN inner-held recordings were exposed in earlier work. Head
seeds are not full forecasting seeds. Upstream exposure prevents a whole-pipeline
OOF or independent generalization claim. Independent selection, calibration and
confirmation remain unopened. The original reference policy also fails this
budget; comparison to it is not a new deployment certificate.

Fresh_run: new policy fitting and fixed-final inference. Cached_verified:
ancestor source packets, original models and separate report arithmetic.
Not_run: new trajectory model, new JEPA/residual, independent testing or deployment.

obs8/pred12 rawstride12,pixel/local detector-silver only. No metric,seconds,
human gold,physical safety,true3D or foundation claim. Stage5C and SMC remain off.

## Next Decisive Experiment

Before another transfer claim, compare whole-recording constraint optimization
against the present64-rows-per-record stochastic loss on original TRAIN. Inspect
constraint residuals and rare-harm gradient coverage while preserving beneficial
interventions; keep the2% budget and complete recording roster. Do not use this
held readout for threshold selection or claim that a longer run is already a fix.

See[complete readout](readout.md),[failure analysis](failure_analysis.md),
[method](protocol.md),[execution amendment](execution_amendment.md) and
[reproduction](reproducibility.md).
