# Original-versus-final selected-score diagnosis

## Role and timing

Registered after both36-head fitting experiments are frozen,while the new
query-unit72-group policy readout runs. Design uses actual training telemetry
and the completed original policy failure,not partial new-unit policy scores.
No new weights,thresholds,choices or risk criteria. All material is exposed
SOURCE fitting/resubstitution. Own diagnostic values are not yet computed.

## Fixed questions

For each old/new unit mode and row-local/selected-set critic,compare score
accounting on the immutable ORIGINAL selected set and on that critic's final
decoded joint set. This gives eight variants in each of the72 original groups.
The final sets are cached_verified frozen choices,not freshly reoptimized.
Re-infer all four critic predictions and require their registered hashes and
choice hashes to match the completed policy readouts before score accounting.

For each complete nonempty selected query and each all/easy channel:

```text
true signed score = sum(true harm - 0.02 * true reference) / sum(causal scale)
pred signed score = sum(pred harm - 0.02 * pred reference) / sum(causal scale)
underprediction = true score - pred score
                = sum(true harm - pred harm) / sum(causal scale)
                  + 0.02 * sum(pred reference - true reference) / sum(causal scale)
```

Report signed-score MSE,each mean component,zero true references,strict
predicted-nonpositive/observed-positive sign patterns,introduced/removed/action-
changed rows,and complete/unknown/empty selected-query counts. A strict sign
pattern is not a ratio-support or recording-safety certificate. All queries
and unknown peers remain;unknown/empty sets receive no invented zero mean.
Only complete-query raw totals are labelled as such.

The original-set MSE,converted through the unchanged original all-action RMS,
must match each existing readout's original-set diagnostic (rtol1e-10,
atol1e-14 for reordered floating arithmetic):288 controls across72 groups.
Within the decomposition require rtol/atol1e-12. These are numerical tolerances,
not relaxed risk thresholds. The original complete-recording2% criterion stays.

## Aggregation and interpretation

Aggregate counts are reused views,not independent rows. For moments,average
supported context/head-seed views within locality,then supported localities.
Report all required/supported views/localities. Missing moments stay not
estimable;do not impute them or call conditional means full-cohort effects.
These are descriptive fitting diagnostics,not new bootstrap hypotheses.

The original versus decoded populations differ. Error differences do not
causally identify selection shift,harm bias or reference bias as the unique
root cause. The decomposition is arithmetic. It cannot prove conditional-mean
misspecification,independent calibration or successful world dynamics.

## Execution

Require the full new72-group readout before computation;bind its complete,
summary and training-freeze hashes in a separate immutable execution receipt.
Independent full-policy replay can continue separately. This diagnostic performs
no optimization or policy intervention. Reuse the original source packets,
supervision and cached choices with strict identities. Native arm64 CPU4,
interop1/workers0,separate lock/log/heartbeat,12h limit. Per-group immutable
results permit interrupted reruns to recompute/check existing groups;completed
runs refuse replacement. Require a separate full72-group diagnostic replay.
Publish aggregate/code/receipts only,never raw labels/row arrays or weights.

No future label/mask enters an inference feature. No independent role opens.
Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates.
No metric,seconds,human-gold,true3D,foundation,submission-ready,Stage5C execution
or SMC claim. Existing deployment remains unchanged.
