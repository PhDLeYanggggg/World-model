# Execution And Verification Receipt

All observations below were made on5November2026. They distinguish engineering
reproduction from scientific success.

| Phase | PID | Observed exit | Duration seconds | Evidence |
|---|---:|---:|---:|---|
| Real pilot | 25011 | 0 | 83.9762 | Two exact50+50/100 comparisons; original masked control exact |
| Full training | 25357 | 0 | 1084.4293 | Six2,000-update models, plus a2,000-update exact refit |
| Development evaluation | 27000 | 0 | 68.8964 | All six fits frozen first; exposed nested-C only |
| First verification | 27153 | 0 | 69.3775 | All six fitting metrics/attributions and development predictions/actions exact |
| Completed read-only replay | 27506 | 0 | 69.3931 | All29then-existing public files byte-identical; no training/new scientific readout |

The complete analysis SHA256 is
`f1b1b4d1caeaf6ff8f314233e5af3ea959582897d15aedb358a3112989ff00f2`.
Training peak RSS was7,617,871,872bytes. The real pilot used500updates including
repeats; primary training used12,000 plus2,000 full-refit updates. Native arm64
Torch CPU4/inter-op1/workers0, atomic100-step checkpoints and heartbeat. No
resource-probing/multiprocessing path or CREATE job was needed.

## Focused Tests

26tests passed in3.43s across rollout-history risk/readout/reporting, paired
risk, joint risk, fixed risk readout, decision semantics and completed replay.
13additional original-rule tests passed in2.01s across selection-forecast
readout, reference-support preflight and policy-conditional readout. After
adding source-influence reporting, its four tests passed in1.41s; three overlap
the original26. Total distinct focused tests:40, not43. The entire historical
suite was not rerun. Real training/evaluation is separate from these tests.

The generated loss/cost figure was viewed as a rendered PNG. Labels, curves and
bars are visible without overlap. The lightweight SVG is the report figure;
private caches, model weights and rendered PNG remain local.

## Scientific Verdict

The internal fitting screen passes3/3, but the complete policy fails original
held risk/support0/17 and worst easy-net degradation is8.660257%. Forty passing
tests, exact replay and a small positive development contrast do not reverse
that verdict. No identity exemption, threshold tuning, source exclusion,
deployment change or independent-confirmation claim. Original24-group primary
remains NA. All prior failures remain failures. Stage5C/SMC remain off.

The first prospective GitHub push returned a server-side commit_refs_rules
error; a retry succeeded and remote commit33b4dd20was verified before formal
training. No training process was duplicated because of that network failure.
