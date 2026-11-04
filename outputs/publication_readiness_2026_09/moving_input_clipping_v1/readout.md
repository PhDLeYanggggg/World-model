# Input Clipping Was Not The Main Supported Explanation

## Material Passport

Fresh_run: all 18 critic-fitting input audits and a separate exact scalar
replay. Cached_verified: original data, normalized inputs, parent weights and
forecast bank. Post-readout diagnosis, not preregistered confirmation. There
are no new optimizer updates, held scores, thresholds, data roles or policies.

## Hypothesis

The common normalizer is fitted only from six physical candidate slots, while
the current intervention learner permits the neural slot alone. If neural
rollout features fall far outside the physical range, hard clipping at +/-8
could remove information needed to distinguish harmful predictions. Measure
that possibility before changing an input map or training another model.

The local-history arm has 380 active inputs; its 48 shared-motion slots are
zeroed. All 18 original heads select regions causally, before target labels
are used for harm-mass accounting. Future masks do not define these regions.
Unknown rows remain in row counts and have separately bounded cost accounting.

## Complete Fitting-Side Result

| Input/causal region | Repeated row views | Rows with no clipped field (%) | Known H mass on clipped rows (%) | Known EH mass on clipped rows (%) |
|---|---:|---:|---:|---:|
| All neural candidates | 1,171,716 | 99.3069 | 6.2665 | 1.2644 |
| Parent independent choices | 803,234 | 99.6067 | 2.0090 | 0.9454 |
| All six physical candidates | 7,030,296 | 99.2792 | 0.3156 | 0.2956 |

The largest absolute standardized feature value is 118.0525, so outliers
exist. But the selected known easy-harm failure is not concentrated in rows
with clipped inputs. Similar row clipping rates appear in the physical bank.
This does not support input clipping as the main explanation for the very
large selected easy-harm underprediction.

These are pooled fitting-view summaries, not IID samples or held test scores.
Clipped rows can affect other rows through query pooling and optimization, so
the mass percentages are not an upper bound on a retraining effect. The audit
also does not establish that physical-only loss RMS or other preprocessing
choices are harmless. No causal root cause is uniquely identified.

## Decision And Execution

Do not launch another clipping-map training sweep from this weak diagnosis.
First check whether each required recording can satisfy the original finite
2% risk contract even with a label-aware fixed-proposal witness. An intrinsic
candidate/support failure cannot be repaired by more critic optimization.
That next audit changes no risk threshold or population.

Evaluation PID 17777 exits 0 in 143.2428 seconds; separate replay PID 17937
exits 0 in 143.6915 seconds. Every scalar matches. Peak replay RSS is
7,610,744,832 bytes. Three focused tests verify original-scale clipping,
unknown-row retention, label-independent causal regions and invalid inputs.
Full arrays and models remain private; `analysis.json` stores aggregates only.

Pixels/native 8/12 steps are not metres or seconds. No new deployment,
physical-safety certificate, true 3D, foundation or submission-ready claim.
Stage5C execution and SMC remain disabled.
