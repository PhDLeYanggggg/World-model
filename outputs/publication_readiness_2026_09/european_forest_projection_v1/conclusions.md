# Forest Projection: A Real Mechanism, Not a Safety Repair

## Result

All 72 frozen forest heads were re-evaluated on their original source-held
validation rows. Prediction and action hashes match the existing receipts.
The control changes only the feasibility decoder, not training or thresholds.
It removes 1,373 of 95,455 selected occurrences. Among the 1,365 removed rows with
known labels, 626 incur positive harm (45.86%, versus 13.47% in the parent selected
pool). This is descriptive enrichment, not a causal effect of changing the
model or independent confirmation. Counts repeat rows across heads/controllers.

| Source diagnostic | Original proportional decoder | Harm-first decoder |
|---|---:|---:|
| Selected occurrences | 95,455 | 94,082 |
| Selected unknown outcomes | 918 | 910 |
| Complete finite-support passes /72 | 33 | 25 |
| Defined selected easy-risk views | 43 | 28 |
| Easy completion-upper violations | 7/43 | 3/28 |
| Worst selected easy completion upper | 5.4058% | 5.6166% |

The four eliminated risk-violating views all become empty selections. No
previously failing source group becomes a complete pass. Eight previously
supported groups become empty and fail. Fewer defined violations are therefore
not a repaired safety gate. The unchanged budget is2% selected positive
harm/selected reference error, distinct from whole-easy net degradation.

## Utility

Conservative utility change: **-0.00011864% of full known reference mass**;
nominal95% locality-bootstrap interval **[-0.00031138%, +0.00001530%]**.
This does not demonstrate an overall utility improvement.

Compared with expected uniform thinning to exactly the same count within each
recording+frame query, the contrast is **+0.00041364%** with nominal95% interval
**[+0.00005418%, +0.00086098%]**. This small development-only discrimination
signal is real for the registered linear expectation comparator. It is not
ADE/FDE lift, the risk of a randomized policy, or a deployment result.

Bootstrap uses3000 draws over12 exposed development localities, averaging
heads/controllers within each locality first. Seven locality means are positive
for the matched contrast, one is negative and four are zero. The three head
seeds share upstream seed43; they are not three independent end-to-end seeds.

## Mechanism and Decision

The original projection reduces predicted harm on23,293/596,988 evaluated
occurrences, including4,077 selected occurrences. It reduces predicted easy
harm on1,957 selected occurrences. All1,373 removed actions were on rows whose
predicted harm had been reduced by the original projection.

Thus decoder-induced risk optimism exists in the current forest. However,
reversing it does not solve conditional-risk prediction, unknown outcomes or
coverage. The worst remaining completion upper increases; the entire-group
success count decreases. **Do not deploy this decoder as a safety repair.**
The original frozen predictor remains unchanged; this experiment creates no
new deployable model and makes no CVPR-readiness promotion.

## Evidence Status

Models/features/splits: `cached_verified`. Inference, control readout and
bootstrap: `fresh_run`, exact full72 replay. New model training, directional
transfer, independent confirmation and full historical test suite: `not_run`.
Fourteen scoped tests pass. Scalar arithmetic checks6624 fields during the
run; a separate aggregate reader checks3210 additional fields. Model inference
itself is replayed using the same implementation, not an independent model.

Boundary: obs8/pred12, raw stride12, image-local detector-silver, previously
exposed European development data. No metric/seconds, human-gold, physical
safety, true3D, foundation or submission-ready claim. Independent roles remain
closed. Stage5C and SMC remain off.
