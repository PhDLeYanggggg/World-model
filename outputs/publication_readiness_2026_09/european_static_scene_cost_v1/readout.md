# Static Scene Context Improves Known Utility, Not Neural Value or Safety

## Actual Training and Fixed Readout

Fresh native arm64Torch,54heads acrosssixcontexts,threecontroller seeds and
threecandidate banks:108,000primary updates,26,309parameters each. Training
PID85320exit0 takes339.605738s,peakRSS10,181,656,576bytes. Every fit matches
its original motion initialization/normalizer,row draws and RNG states.
The1,152new projection weights begin at zero and actually update in all fits.
The frozen trajectory producer is not retrained. Old54fits are cached-verified
controls,not another54fresh models.

The real120-update recovery pilot is exact across all banks. The full2,000-
update disabled branch reproduces old network weights,Adam moments,draws and
loss traces exactly. All54new endpoints commit at `755bdd7d` before readout.
First readout PID86269exit0 completes144views in205.965584s,
peakRSS7,669,972,992bytes. Exact replay is recorded separately in
`verification.json`;aggregate archives preserve every numerical JSON byte.
No independent role is opened.

Separate exact replay PID86906exit0 takes207.692464s. The final focused suite
passes31tests in2.44s;the full historical suite was not rerun. The additional
camera-mapping fixture initially used an invalid filename suffix and was fixed
without changing model code or weights. The evidence exporter initially assumed
all historical aggregates were bundled;it now verifies either immutable format
without repacking old studies. Both document exports check exactly.

## Known-Label Utility

Equal-locality known-label ADE gain over the same causal floor:

| Bank | Gain | Conditional95%locality CI |
|---|---:|---:|
| Static-context physical-only | 1.076598% | [0.258362,2.522915]% |
| Static-context physical+neural | 0.982769% | [0.242283,2.301831]% |
| Static-context neural-only | 0.433381% | [0.206401,0.696722]% |

The mixed scene-minus-no-scene difference is **+0.428982pp**,
CI[+0.053706,+1.044617]. Same-count utility difference is+0.112189pp,
CI[+0.028429,+0.232094]. This is useful exploratory evidence that static image-
region position contains information absent from relative-motion cost features.
It is not an independently established world-model contribution.

The effect is heterogeneous:seven locality point gains are positive,four
negative,one zero. Locality020has+3.339750pp and contributes substantially to
the equal-locality average. No locality is removed or used to select deployment.
The three new mixed head-seed gains over the floor are0.954616%,1.044915%and
0.948777%;these are controller seeds,not three fresh trajectory producers.

The normalized harm-MSE change is-0.106608,conditional interval
[-0.173896,-0.047766]. This is a squared-error difference,not a percentage.
It comes from a fresh descriptive bootstrap of cached-verified aggregate
moments;other moment intervals and all original utility intervals are retained
in the[new six-study evidence](../feature_cost_evidence_20261029.md).
Lower overall harm-MSE does not establish accuracy on the selected tail.

## Neural Increment Still Fails

Mixed minus physical-only is **-0.093829pp**,
CI[-0.217875,-0.003382]. Completion-lower difference is-1.195683pp,
CI[-3.282407,-0.065640]. Supplying the neural candidate still does not establish
value beyond the physical candidate bank on this development comparison.

For the scene change itself,the completion-lower difference versus the old
mixed policy is-0.405680pp,CI[-1.185725,+0.126229]. Its matched-count
completion-lower difference is-0.026936pp,CI[-0.129528,+0.055094]. Therefore the
positive known-label result is not a positive worst-completion result.
Unknown outcomes cannot be imputed as favorable or silently dropped.

## Safety Still Fails

Primary mixed policy supports **23/186** required recording/head views and
12/72source/head views. It selects32,219row views,including518unknown outcomes.
There are34known all-risk violations,96known easy-risk violations and20undefined
easy-risk views. Fitting support is77/444views,not a fitted safe policy.

Fixed4xharm sensitivity supports80/186views,with44undefined easy views;
it is not promoted or used as a held-selected threshold. Count matching is
utility-only. Empty selected risk remains `not_estimable`,never safe zero.
Known utility,lower MSE,static context and extra parameter capacity cannot
override the unchanged every-required-recording all/easy2%contract.

## Interpretation and Next Action

Reference sharing repaired a measured inconsistency without improving the
policy. Static context now improves known-label routing,but more than physical
motion features is not the same as more useful neural forecasting. Broadly
increasing loss weight or model width is not justified by these results.

The next targeted question is **selection-conditioned risk calibration**:
does choosing the maximum predicted gain among candidates concentrate harm
underprediction or reference overprediction? Audit that on frozen fitting
outputs,contrasting all eligible actions with actually selected actions,without
changing thresholds or opening reserved roles. Then choose a calibrated
learning repair only if that mechanism is measured. The forthcoming moment-
error export is descriptive,not a guarantee about that selected tail.

The older partial-neighbor and track-preserving neural refits are not new
solutions:the first failed its incremental screen,the latter gave a small
matched forecast gain but not safe CV-relative easy behavior. Preserve those
results rather than repeating a generic context/mask experiment.

## Scope and Reproduction

Obs8/pred12/rawstride12,EuropeanSquares SOURCE webcam detector tracks,raw pixel
box midpoints. Static polygons are place-region proxies,not walkability,
obstacles,verified viewports or exit goals. No future labels/availability enter
features;no central velocity,test statistics or endpoint-built goals are used.
Exact raw-camera mapping preserves distinct cameras within merged localities.

SOURCE inner-held recordings were previously exposed. The3,000locality
bootstrap is conditional exploratory evidence,not independent confirmation or
whole-pipeline OOF. Repeated context/head views are not independent recordings.
The corrected extra mapping fixture and current partial-context invariance
checks are regression tests,not new scientific samples. No deployment,metric/
seconds,true3D,foundation,Stage5C orSMC claim/execution follows.

```bash
.venv-pytorch/bin/python scripts/run_m3w_static_scene_cost.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_static_scene_cost_v1 --verify
```

Fresh phases are `register`, `pilot`, `neutral`, `train`, `evaluate`. Completed
receipts are immutable;interruptions use `--resume` with the same fixed roster.
Private licensed source arrays/checkpoints are required. Public aggregate replay
alone is not end-to-end raw-data reproduction. See[protocol](protocol.md),
[training freeze](training_freeze.json),[summary](summary.json) and[model card](model_card.md).
