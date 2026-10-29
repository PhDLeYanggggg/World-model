# Published Regression-Deferral Control: Accuracy Without Required Risk Support

Fresh_run primary training and inference,29October2026. All54nativeTorch fits
and108,000updates completed with matched source row draws. Every endpoint was
frozen in d5ad1e3d before first readout. No independent evaluation role opened.

## Execution and Identity

- Training PID65672,exit0:200.797506s,peakRSS11,402,346,496B.
- Each conditional loss head:24,961parameters;six frozen contexts,three seeds,
  physical-only,physical+frozen-neural and frozen-neural-only banks.
- First readout PID69045,exit0:211.457225s,peakRSS8,055,062,528B.
- Exact inference/aggregate replay PID70427,exit0:206.839815s;all144groups
  and summary match exactly. Original execution receipts remain unchanged.
- Individual aggregate JSON bytes are preserved in the2,013,081-byte metric
  archive,not raw data or weights. Packaging verification is not a fresh
  scientific computation.

## Fixed Inner-Held Results

These are exposed internal development records,not an independent test.
The known-label ADE gain is over the same causal floor,equal-weighted by
locality after averaging context/head-seed views.

| Policy | Known ADE gain over floor % | Full risk-supported record/head views |
|---|---:|---:|
| Physical-only published loss |7.229477|0/186|
| Physical+neural published loss |8.018405|0/186|
| Neural-only published loss |7.338229|0/186|
| Physical-only,fixed cost gate |0.479507|40/186|
| Physical+neural,fixed cost gate |0.472543|30/186|
| Neural-only,fixed cost gate |0.288656|33/186|

The mixed policy's gain interval is[5.042421,10.822614]%. Its head-seed gains
are8.092842,6.871677and9.090695%. These are three newly trained deferral heads,
not three new forecasters. There are31unique inner-held source/recordings;
186repeated recording/head views must not be treated as186independent videos.

## Is the Frozen Neural Expert Adding Value?

Compared with physical-only,the mixed bank has a known-label paired gain of
+0.788928pp,conditional3,000-locality bootstrap CI[-0.170983,+1.675128].
The interval includes zero. With unknown selected outcomes included through
causal disagreement completion bounds,the paired lower contrast is-4.463329pp,
CI[-10.643677,-1.085987];the paired upper is+6.041185pp. These are completion
bounds,not observed negative/positive outcomes for missing labels.

At matched counts,the known-label contrast is+0.893883pp,CI[+0.162533,+1.689813],
but the matched completion-lower contrast is-3.156553pp,CI[-7.664351,-0.605739].
Count matching tests utility only. It does not turn an unsafe policy into a
passed safety gate or establish a robust neural increment under unknown labels.

## Why This Is Not Deployable

The unrestricted mixed policy makes381,203interventions over repeated row
occurrences,including8,144unknown-outcome rows. Known all-risk is violated in
185/186record/head views,and known easy-risk in183/186. This is not the empty
policy failure of the earlier recording-aware head: it switches extensively
and harms too many required slices. Average accuracy alone misses that harm.

The fixed predicted-cost gate reduces interventions to23,400,including309
unknown rows. It improves risk support to30/186but leaves63known easy-risk
violations and23undefined easy-risk views. It is a preregistered control,
not a selected threshold,calibrated certificate or deployable repair.

Fitting known-label gain is21.735065%for the mixed arm versus8.018405%on exposed
inner-held records. That descriptive gap is compatible with overfitting or
distribution shift;it does not uniquely identify a cause. Full fitting risk
support is0/444even before transfer. The unconstrained objective itself is
not aligned with the required selected positive-harm ratios.

## Interpretation and Next Controlled Test

This applies Mao etal.,ICML2024,Section4Eq.(3),using continuous expert costs,
not one-hot oracle labels. It neither reproduces the original experiments
nor verifies the paper's bounded-loss theorem assumptions for this native
partial-mask ADE application. It adds a meaningful public-loss control,but
does not pass the original2%recording requirement.

The parallel support-unit repair tests the measured collapse of the distinct
recording-constrained head. It does not change this failed result,its weights,
the risk contract,or the independent-role boundary. No new neural trajectory
producer,scene-joint optimizer,deployment,metric/effective-second,true3D,
foundation,Stage5C execution orSMC result follows from this control.

Primary source: [Regression with Multi-Expert Deferral,ICML2024](https://proceedings.mlr.press/v235/mao24d.html).

## Reproduction

From the repository root,using the native arm64 environment:

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_regression_deferral_v1 --restore
.venv-pytorch/bin/python scripts/run_m3w_regression_deferral.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_regression_deferral_v1 --verify
```

The first command restores only missing owned aggregate files byte-for-byte.
The second requires the private source assets and frozen checkpoints;the public
metric archive alone is not sufficient for fresh inference or model training.
