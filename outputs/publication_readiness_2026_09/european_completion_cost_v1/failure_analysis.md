# Why Completion-Aware Cost Supervision Failed

## Result And Scope

This is a completed, negative fitting experiment, not an independent test.
All 72 fresh Torch heads, 144 primary readout groups and 888 paired fixed-region
diagnostic views complete. Separate reconstructions match exactly. The frozen
forecast bank, physical-fitting normalizers, causal query draws, optimizer,
2,000-update budget and initialization are matched within each training pair.
Only the inclusion of unknown-row interval-endpoint losses changes.

The mixed policy's known-label ADE gain changes from +0.580799% to -18.512092%.
The paired change is -19.092891 percentage points, fitting locality-bootstrap
95% CI [-50.819548, -1.021775]. Full required recording-risk support falls from
77 to 26 of 444 dependent views. All four scientific advancement checks fail.
Neither completion training nor the fresh known-only control is deployable.

## Failure Taxonomy

### 1. Endpoint Proxy Fit Is Not Conditional Risk Calibration

For missing outcomes the extra targets are conservative interval endpoints,
not newly observed errors. In the mixed bank, pooled unknown-proxy harm MSE
decreases from 51.8355 to 33.1353 and easy-harm MSE from 74.7285 to 56.7123.
However, known-label harm MSE increases from 0.506626 to 1.124881 and easy-harm
MSE from 0.023455 to 0.226372. Overall easy-harm proxy MSE decreases slightly,
1.596060 to 1.415449, while overall harm MSE increases, 1.587144 to 1.798728.
Thus even the pooled error changes are mixed. These normalized occurrence-
weighted statistics are not loss-comparable across targets or calibration
certificates. Unknown outcomes remain NaN throughout the primary evaluation.

### 2. The New Policy Moves Into A Badly Estimated Region

Both frozen estimators are scored on identical legal actions and selections.
For the mixed bank, the median predicted/observed easy-harm ratio is:

| Fixed region | Known-only estimator | Completion estimator | Finite / undefined views |
|---|---:|---:|---:|
| All legal actions | 1.152863 | 4.710467 | 444 / 0 |
| Known-only policy selection | 0.019880 | 1.652333 | 328 / 116 |
| Completion policy selection | 0.564438 | 0.000628 | 418 / 26 |

Each median is conditional on a strictly positive observed harm denominator;
its finite count differs from the easy-reference support count used by the
full policy gate. Undefined views are retained and are never called safe.
The completion model improves this descriptive ratio on the old fixed region,
yet severely underpredicts easy harm in its own newly selected region. Its
corresponding total-harm ratio there is 0.075503. Global easy-harm
overprediction therefore does not prevent selection-conditioned failure.
The intervention count rises from 80,414 to 237,281 repeated row views,
including 932 versus 3,449 unknown-outcome selections. Known all/easy risk
violations rise from 85/234 to 311/375. Unknown errors are not asserted to be
harmful simply because their upper bounds are large.

This is evidence of risk-estimation mismatch on the induced selection region,
not a unique causal proof that a particular architecture, gradient effect or
missingness mechanism caused the failure. The new loss also changes reference,
benefit and action-ranking fits; the comparison does not isolate each effect.

### 3. The Negative Mean Is Not Merely A Tiny Denominator

Locality 082 is a large adverse slice. Its unchanged floor mean ADE is
21.838344 pixels; the completion mixed policy reaches 60.963796 pixels, while
the known-only policy remains at the floor. This is native error growth, not
only an exploding relative ratio. Other negative localities also remain in
the primary aggregate. These values average repeated context/head views and
are not independent sample counts. No locality is removed or reweighted.

### 4. A Pure Veto Cannot Repair Missing Selected Support

The known-only mixed policy has 62/444 views with undefined selected easy
risk. With the fixed floor and easy event, actual easy-reference mass is
nonnegative and independent of which candidate is chosen on a retained row.
A policy restricted to those selected rows cannot turn zero support into
positive support. Its defined-support ceiling is at most 382/444, before
considering any known violations or completion bounds.

The same nesting holds for an availability-probability veto that leaves the
conditional heads fixed. See the [restricted-policy argument](support_ceiling_note.md)
and its executable checks. This is not an impossibility result for new
conditional heads, inputs, forecasters or M3W generally. It is a reason not
to spend another training grid on that particular patch.

## What This Experiment Does Not Establish

- It does not identify missing-label exclusion as the sole original failure.
- It does not establish irreducible uncertainty or information insufficiency.
- It does not prove that larger models, better inputs or new labels cannot help.
- It does not evaluate an independent scene, whole-pipeline OOF or deployment.
- It does not demonstrate new neural trajectory or world-dynamics capability.

## Next Decision

Do not tune another alpha or reject-only threshold on this readout. First
compare a proposed repair with the existing task-risk, easy-harm, reference
and region-moment objectives; renaming an old loss is not a new experiment.
A useful next intervention must address conditional risk on causally
admissible expanded regions or add lawful predictive information. Any new
labels must retain their provenance and role boundaries. Future-label easy
anchors, relaxed risk thresholds, omitted recordings and held-set tuning are
not repairs. Independent roles and DroneCrowd remain closed.

SOURCE is oblique detector-track midpoint pixel data, obs8/pred12 with
stride12 raw frames. No human-gold, metric, effective-second, true-3D or
foundation claim is supported. No Stage5C or SMC is executed.
