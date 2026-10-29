# Full TRAIN Census: Fitting Improves, Generalization Does Not

## Evidence

Fresh_run: all 288 frozen-head full-TRAIN inferences completed in
114.16s. Models, source packets and previous development
aggregates are cached_verified, not freshly trained or newly evaluated.
There are 72 matched identities per arm, not 288 independent datasets.
All TRAIN rows, including unknown outcomes, remain in inference. No target is
an input or action filter. Repeated predictions and the 2,000-step sampling
histories replay exactly. No optimizer, new development inference, independent
selection/calibration/confirmation access or deployment change occurred.

| Frozen arm | TRAIN known easy violations /72 | Development known easy violations /72 | TRAIN pass, development fail /72 | TRAIN completion support /72 |
|---|---:|---:|---:|---:|
| quadratic | 58 | 60 | 7 | 0 |
| task_utility | 64 | 62 | 3 | 0 |
| unit_control | 58 | 58 | 7 | 1 |
| budget_scaled | 2 | 58 | 56 | 2 |

The unchanged 2% risk is positive selected harm/reference error, not net ADE.
The scaled policy has 2/72 known easy violations on full TRAIN, compared with
58/72 for its unit control; it still has 58/72 on held-recording development.
56/72 scaled views pass the known TRAIN calculation but fail on development.
This supports prioritizing out-of-recording harm prediction over another global
penalty increase. It does not identify a unique causal mechanism or restore
independence to exposed development results.

## Unknown Outcomes Still Block Safety

The scaled model has 66/72 TRAIN completion-upper easy violations, four undefined
known easy ratios, and only 2/72 full finite-completion supports. In
64/72 views a known-label easy
calculation passes but its completion-upper easy calculation fails. Unknown
outcomes cannot be discarded or treated as zero harm. Full TRAIN fitting success
on known labels is not a safety guarantee, nor evidence of new world dynamics.

## Coverage and Rare Harm

Across the 72 matched sampling identities, 12
leave at least one known TRAIN query unseen in 2,000 updates. Median unseen-query
fraction is 0.000000. Median unseen easy-harm
mass fraction is 0.000000; its maximum
is 0.062519. The largest 1% of TRAIN
row easy-harm values contain a median
0.702202 of its total easy-harm mass.
These coverage statistics do not establish that unseen rows caused development
failure. The census includes 1,316,826 repeated
TRAIN row occurrences per arm; overlapping views are not independent samples.

## Next Repair Boundary

Keep existing fits and failed development results frozen. Any next change must
test harm generalization inside TRAIN using whole recordings, rather than tune
against these already-exposed development labels. Check available recording
support first, retain unknown-label completion accounting, and compare against
the existing negative sampling/calibration controls before commissioning fits.
Increasing model size or repeating the same threshold search is not justified
by this census. The separate prospective contract repair prevents impossible
same-count safety requirements; it does not rescue the current unsafe model.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_train_census --check
```

This verifies all receipts, registered source hashes, checkpoint bindings,
sampling equality, development aggregates, row totals and risk mass ratios.
No new inference is needed for report verification. Observation8/prediction12,
dataset-local coordinates and raw-frame conventions are unchanged. Not true3D,
not a foundation model, no metric/seconds claim; Stage5C and SMC remain disabled.
