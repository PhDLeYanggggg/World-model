# Query-unit risk critic: model and data card

## Purpose and status

Test one training-unit correction in a controller over seven frozen forecasts.
This trains risk critics,not new trajectory dynamics. Known fitting utility
improves,but the primary robust-utility/risk requirement fails. The model is
not deployed and is not an independent or submission-ready world-model result.

## Model

Two equal-size critic families,row-local and original-selected-context,
30,726 parameters each. Six contexts and seeds17/29/43 yield36 fresh heads;
36 preceding frozen heads supply cached_verified controls. The same462 causal
fields,five original moments,action indicator,envelope and count metadata are
used. Original selected-set context remains deterministic past-derived input,
not a future label or a new observation.

Only positive harm and easy-harm outputs change through the parent's bounded
anchor-exact link. Benefit,reference,easy-reference and all forecast paths stay
fixed. These bounds are model constraints,not physical-safety guarantees.
The new query RMS is a training-label loss scale,not an inference field.

## Data and access

EuropeanSquares oblique webcam detector-track development in local image
coordinates:149,510 unique target rows,18,766 current-frame queries,74 recordings
and12 exposed localities. Reused context/seed views total897,060 rows and112,596
queries. They are not additional independent samples or SDD evaluation rows.
Task:obs8/pred12/rawstride12,not historical raw-frame t+50.

Inference accepts declared causal inputs only. No future endpoint,central
velocity,future remaining track length,oracle choice or test-endpoint goal is
an input. Future labels/support masks serve loss/evaluation only. Unknown rows
remain in draws,context and full-policy evaluation. This input contract does
not establish whole-pipeline OOF,producer independence or online causal perception.

Labels are detector-silver,not human gold. ADE uses available future points;
missing required endpoints prevent full-roster FDE. Wholly-unknown completion
bounds do not certify missing interior labels. No metric/time calibration.

## Actual computation

Native arm64 Torch CPU4/interop1/workers0. Real600-update pilot:21.238s,
6.468GB peak RSS,exact old-control reproduction and interrupted recovery.
Complete36-head fitting:299.778s,10.451GB peak,72,000 updates plus4,000 separate
full-refit updates. Both first2,000-step refits agree scientifically.
Full72-group readout:1746.472s,11.026GB peak. Separate complete replay:
1788.801s,10.378GB peak. All72 groups/144 fixed-critic controls agree exactly.
No validation/test checkpoint selection or threshold search;final step is fixed.

## Performance and restrictions

Known ADE gain6.574842%->8.215347%,paired+1.640505pp
CI[+1.101933,+2.251535]. Primary completion-lower change-2.597889pp
CI[-7.642133,+0.293356];complete risk1/444 supported,383 known easy violations.
The corrected-units context arm still loses to its local control. Original-set
query MSE improves but final-selection errors and missing support remain.

No fresh independent scene,SDD-safety transfer or new physical-only bank control
with these heads is evaluated. The prior physical-bank result is a different
fixed-critic experiment. Do not attribute all known gain to learned dynamics.
No independent efficacy,calibrated safety,true3D,foundation,metric/seconds or
deployment claim. Stage5C execution/SMC remain off.

## Artifacts

Public:registered code/config,receipts,training telemetry,aggregate metrics and
self-generated figures. Private:source packets,feature/history/latent caches,
per-row choices and `.pt.gz` checkpoints. These are not included in Git.
Aggregate reconstruction is distinct from full raw-data/weight reproduction;
the latter requires the registered local assets. See reproduction.md.
