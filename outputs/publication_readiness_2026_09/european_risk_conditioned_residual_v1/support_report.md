# Fitting-Only Producer Diagnostic

Fresh144 frozen-Torch fitting predictions;432 exact fixed-design producer projections.
This is not new neural training, causal attribution or held-out accuracy.

| Inputs/bank | Views | Median score-gap RMS | Minimum easy-harm rows | Median outer outside inner central90% (two risk scores) |
|---|---:|---:|---:|---|
| full/oof | 72 | 0.0774954 | 245 | [0.07281506507177779, 0.11471269580628689] |
| full/in_sample_next | 72 | 0.0512957 | 245 | [0.0892525851452746, 0.10507025200996112] |
| full/in_sample_prev | 72 | 0.0622179 | 245 | [0.12314350374864888, 0.11437272629789397] |
| motion_only/oof | 72 | 0.02926 | 24 | [0.08002264235957386, 0.14381757502948384] |
| motion_only/in_sample_next | 72 | 0.0309518 | 24 | [0.08108758228792445, 0.0819700448623442] |
| motion_only/in_sample_prev | 72 | 0.0244269 | 24 | [0.1156311439199019, 0.10465390120807377] |

Training allowed: True. All views retained. Central90% is descriptive, not an OOD or safety bound.
Original versus inner scores differ on matched fitting rows. Identity isolates a producer-dependent term but does not establish the source of held errors.
No fitting support threshold was selected from held outcomes. Obs8/pred12 annotation steps, pixels only.
