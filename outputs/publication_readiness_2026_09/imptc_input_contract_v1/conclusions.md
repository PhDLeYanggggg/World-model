# IMPTC Input Contract: A Measured Unit-Dependence Failure

## What Was Actually Run

This is an engineering-input experiment, not an external forecast benchmark.
On 58 fixed-modulus queries from all four sample recordings, 754 past-eligible
agent windows were passed through two real frozen Torch predictors and three
frozen risk heads. Each identical history was represented at .01x, 1x and 100x
coordinate units. Future coordinates, endpoint errors, tracker status and
whole-track classes were not inference inputs or selection criteria.

The query/agent count is not an independent statistical sample size. The source
has one physical intersection and mixed road-user types. The fixed seed17,
coupa-excluded predictors are numerical probes, not a selected deployment model.
Source master-index obs8/pred12 stride1 is not matched physical time to SDD's
primary obs8/pred12 stride12. No seconds, metric or generalization claim follows.

## Mechanism Result

The old cost feature vector contains 354 normalized columns followed by
`log(native_past_scale)` and `log1p(native_forecast_disagreement)`. A controlled
check held all 354 columns and the normalized candidate forecast fixed, changing
only those last two columns. Multiplying the nominal coordinate unit by 100
changed the predicted net-gain sign as follows:

| Frozen head | Changed signs / 754 |
|---|---:|
| Damped velocity | 411 |
| Transformer | 369 |
| EqMotion | 325 |

At .01x, the corresponding isolated counts are 0, 0 and 1. In the complete old
input pipeline at .01x, Transformer/EqMotion signs instead change in 124/176
windows: the fixed native normalization floor also changes input geometry.
Neither sign counts nor moment changes are observed forecast gains or complete
scene-level policy interventions. See [mechanism arithmetic](cost_unit_mechanism_check.json).

This establishes unit sensitivity of the *current frozen input/head chain*, not
the cause or magnitude of every historical Stage31-40 transfer failure. Those
older stages have additional lineage and selection limitations.

## Repair and Remaining Failure

A separate, non-deployed adapter defines a shared scene unit from the current
eight-step prefix extent. Nearest-neighbor ordering uses dimensionless distance
tie buckets. A new 355-column cost contract removes the native-scale feature
and uses normalized disagreement. It rejects old cost checkpoints: it requires
a new source-only fit with normalized targets and source-only preprocessing.
No frozen code, weights, thresholds or existing scientific protocol changed.

Two synthetic runs first exposed equidistant-neighbor ordering sensitivity under
translation/unit conversion. A deterministic dimensionless tie rule repaired
that issue before the registered actual-data run. It is not an outcome-tuned
forecasting change. The final tests retain this symmetric-neighbor case.

The registered normalized-output tolerance of 1e-5 passes for Transformer and
damping in all 754 windows. It **fails for EqMotion in 3 windows** at .01x:
maximum normalized difference 0.0001175404. Those rows have tiny float32 input
rounding differences (up to 0.00000190735) amplified by the frozen EqMotion core.
The stricter registered gate remains false; it was not relaxed after inspection.

Normalized coordinates alone can exaggerate the apparent trajectory difference
when the normalizer changes. A separate inverse-transform calculation compares
forecasts after restoring both to the original source coordinate unit:

| Pipeline / .01x comparison | Damping max difference | Transformer max difference | EqMotion max difference |
|---|---:|---:|---:|
| Old | 4.66e-9 | 0.0134585 | 0.000632577 |
| New input contract | 1.46e-13 | 7.11e-9 | 0.00000671399 |

These are differences **between two predictions**, not ADE/FDE against truth.
The smaller restored difference does not reverse the normalized-gate failure.
All 2,262 independent damping formula checks pass. Floating-point equivalence
on these factors is not a proof of invariance for arbitrary coordinates.

## Producer Lineage and Evidence Status

All 2,937 bindings in the current guarded producer identity were re-hashed
(6,106,808,709 bytes). Every one of 12 native-coordinate Transformer, 12 EqMotion
and 36 net-moment fit receipts/checkpoints was checked. Their recorded fitting
sites are the excluded-site subsets of coupa/deathCircle/gates/hyang, backed by
33 SDD recordings and 175,756 windows. Bound raw sources are SDD annotations;
no IMPTC/VRU raw path appears in this verified local chain.

This narrows direct-fit exposure concerns; it does not certify remote history,
absence of indirect design exposure, or independence from the related
Aschaffenburg VRU/DeCoInt site. Byte hashing bound label artifacts is not a new
numerical error readout. Upstream IMM processing remains unverified for online
causality. IMPTC is now explicitly covariate-design-exposed by this engineering
work and is not admitted as untouched confirmation or independent calibration.

- `fresh_run`: fixed-prefix input construction, frozen-model forwards, unit
  contrasts, separate inverse arithmetic and isolated two-column control.
- `cached_verified`: prior source conversion and frozen producer artifacts;
  exact rerun of all new probe arrays and aggregate analysis.
- `not_run`: new head/predictor training, external forecast-error evaluation,
  independent calibration, confirmation, deployment and predictive-lift testing.

64 scoped tests pass. This includes eight new input-contract tests and existing
prefix/intake/forecaster tests. The unchanged full legacy suite was not rerun.
The actual run took 26.641 seconds and its full replay 26.582 seconds; these
times include provenance hashing and forward probes, not training. Separate
mechanism/inverse checks have their own outputs. Private probe files total
29,504,157 bytes and are not committed.

## Next Falsifiable Step

First preserve this unit-contract failure and require it at any cross-dataset
entry point. Test a numerical-stability repair for EqMotion without using future
errors or selecting checkpoints. Then implement the same new contract on the
already design-exposed SDD source population at its unchanged stride12, and fit
matched new risk heads with normalized targets. Compare against the frozen head
on source-only development data before any independent-source calibration.

Do not feed the new vector to an old pixel-trained head or reinterpret its easy
cutoff as external-coordinate risk. Source admission still needs an explicit
offline-label observation claim, time-grid contract and independent-site design.
DroneCrowd remains closed. Deployment is unchanged; Stage5C and SMC remain off.
M3W is still a 2.5D trajectory-state research model, not true 3D, foundation
success, verified physical safety or a demonstrated submission-ready method.
