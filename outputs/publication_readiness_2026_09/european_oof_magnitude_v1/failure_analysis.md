# Failure Analysis

Result source: fresh_run source-development readout, cached_verified parent
forecasts/outer heads. No new policy or independent evaluation was run.

| Failure category | Observed evidence | What this does not prove |
|---|---|---|
| Non-uniform task information | True vs both matched controls has only 2/6 positive full-input cost intervals | Not proof that auxiliary representation has zero information |
| Generic shrinkage | Cost-only and shuffled arms obtain similar scaled-vs-raw MSE effects | Not an auxiliary or JEPA contribution |
| Loss/guard mismatch | True scaling improves some MSE comparisons but worsens coverage-log error in 4/6 and top10 capture in 1/6 | Not safe deployment; MSE and aggregate harm mass are different quantities |
| Producer-size transport | One-site references produce two-site auxiliary labels; readout then acts on three-site outer heads | Not a controlled demonstration that more training sites cause the failure |
| Target-definition drift | Easy membership differs by 0.137%-15.012% across inner/outer cuts; median 2.525% | Not permission to relabel held outcomes or replace the approved endpoint |
| Sparse fitting events | True inner cap-event positives range 67-3794 full, 5-1732 motion; some single-site references have zero positive easy harm | Numerical fit support does not establish statistical power |
| Small independent-unit count | Four held localities per assignment; assignments overlap | Thousands of windows or bootstrap draws are not thousands of independent scenes |

## Tested Explanations That Are Not Supported

- A computation timeout is not the cause: all fixed budgets completed normally.
- Invalid costs treated as zeros are not the cause in this run: unknown draws=0.
- Slope-cap saturation is not observed: no fitted component reaches0 or8.
- Failure to optimize anything is not observed: fixed-batch cost loss decreases
  from first to last recorded snapshot in424/432 true and424/432 shuffled heads.
  This is training evidence, not a generalization or convergence proof.
- A universal lack of ranking information is contradicted by the positive
  AUROC comparisons, but ranking alone remains insufficient for cost control.

## Direction of the Magnitude Problem

Full-input true easy-harm slopes range0.003536-1.260776, median0.318410;
motion-only range0.000841-1.479162, median0.060182. These train-fitted slopes
shrink many predictions. Lower squared error can coexist with worse agreement
between total predicted and observed harm. The measured coverage guard fails;
this study does not identify a universally correct adjustment direction or
license selecting an alternative loss/scale from the outer results.

All six assignments and both families remain in the report. Negative point
estimates with intervals crossing zero are unresolved, not proven harm and not
proven preservation. Full and motion-only cannot be treated as a matched
scene/interaction ablation because their forecast populations differ.

## Consequence

Do not deploy this readout or replace trajectory performance claims with cost
MSE improvements. Do not reopen independent roles. Isolate producer-size and
target-definition effects with fixed, source-only matched controls before
continuing a generic auxiliary-label or threshold search. A further failure
must remain visible and may require narrowing the method claim.

Obs8/pred12 native annotation steps, detector pixels. No metric, seconds,
human-gold, true3D, foundation or physical-safety claim. Stage5C/SMC off.
