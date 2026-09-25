# Easy-Harm Exposure Did Not Repair Conditional Risk

## Verdict
The fixed-budget importance-corrected sampling repair failed. No deployment,
risk certificate, new neural-dynamics contribution or submission-ready claim
is promoted. Joint allocation remains a useful source-development signal, but
it is not reliable risk control.

Fresh_run: 36 native-arm64 Torch heads, 72,000 updates, 180 new policy views and
source-C reductions. Cached_verified: forecasts, utility heads and 324 control
views. Registration `49124cf1` preceded training; freeze `8bb149e6` preceded C
readout. The 100-step pilot resumed within the first head's budget. No unknown
future label was sampled. No favorable seed or threshold was selected.

## Registered Primary Comparison
For the full forecast pair, corrected-joint versus uniform mean-joint changes
three-seed all-ADE by **-0.1691% to +0.4865%** across six A/B assignments:
two locality-bootstrap intervals positive, two negative, two overlapping zero.
Hard ADE changes **-0.1782% to +0.4004%**, with one positive interval, two
negative and three overlaps. This is not a stable sampling improvement.

| Full joint policy | Complete observed risk /18 | Net easy /18 | Easy-positive-harm violations /72 dependent locality views | All-positive-harm violations /72 |
|---|---:|---:|---:|---:|
| Uniform mean control | 6 | 18 | 15 | 2 |
| Corrected sampling | 4 | 18 | 21 | 2 |

The easy average can improve while selected easy agents suffer excessive
positive harm. The corrected policy passes net easy in 18/18 settings but
fails the complete conditional-risk requirement in 14/18. Its intervention
fraction is 30.14%-43.59%. Motion-only corrected-joint also loses complete
risk passes, from 12/18 to 8/18; worst net easy degradation is 1.831%, below
2% but not a positive-harm certificate.

## Positive Signal, With Limits
Full corrected-joint versus corrected-dual improves all ADE **0.8446%-2.6287%**
and hard ADE **0.8960%-3.4332%**, all six intervals positive for each. Against
the query-count-matched hash control, all-ADE gain is **0.0948%-0.6227%**, with
six positive intervals. This supports structured allocation over individual
vetoes and count-only selection on these source views.

It does not establish a benefit at equal realized risk: the hash control
matches intervention counts, not observed harm. Against the old raw-neural
rule, full corrected-joint has only one positive interval, two negative and
three overlaps. No broad superiority over the strong prior rule is shown.

## What the Controlled Repair Taught Us
Positive easy-harm exposure rose from 1.85%-3.33% of uniform full-pair draws
to an observed 50.82%-51.63%. Weights preserve the expected mean objective.
The failure is therefore not explained by the sampler simply failing to show
the rare positives to the network.

Whole-B easy-harm component MSE improves in 13/18 full fits (median ratio
0.99394) and 18/18 motion fits (0.95746). However, both reference-mass
components worsen in every fit for both pairs; full all-harm MSE worsens in
18/18. On C, easy-harm MSE improves in only 4/18 full and 3/18 motion views.
The [matched-action diagnosis](fitting_transport_analysis.md) separates this
transport failure from changes in which agents are selected.

## Evidence Boundary
Three seeds are averaged within locality before 3,000 resamples of four C
localities. Source roles overlap and C has historical development exposure;
these are not six independent discoveries or independent confirmation.
The six opened selection localities are not evaluated this round; 12 reserved
calibration and six confirmation localities remain closed. The 2% constraints
are unchanged. Full legacy tests are not claimed; the scoped replay/test
receipt is in verification.json.

Obs8/pred12 annotation steps, raw stride12, image pixels, detector-derived
labels. No metric, seconds-level, human-gold, physical-safety, true3D or
foundation claim. Stage5C and SMC remain off.
