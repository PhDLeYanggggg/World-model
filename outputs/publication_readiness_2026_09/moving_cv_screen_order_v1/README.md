# Screening Before Trajectory Nomination

I kept the previously trained gain, risk and pair heads fixed to test
whether the shortlist itself was the bottleneck. On the same queries,
I compared the original gain-first nomination with two ways of letting
the risk score see more trajectories before choosing: adding full OLS4
to the original nominee, or keeping the entire menu. I also tested the
frozen pair term after full screening. The latter changes recipients,
not the nominated trajectory of an individual recipient.

| Held fitting-source strategy | Edits | Original 2% passes /72 | Undefined easy /72 | Worst signed easy change |
|---|---:|---:|---:|---:|
| Gain-only, matched | 2,317 | 1 | 39 | +54.41% |
| Nominate first, then screen | 2,317 | 13 | 37 | +35.41% |
| Keep full OLS4 too, matched | 2,317 | 10 | 37 | +22.08% |
| Screen full menu first, matched | 2,317 | 9 | 39 | +32.32% |
| Screen first, couple recipients, matched | 2,317 | 11 | 39 | +18.99% |
| Screen full menu first, use available cap | 5,337 | 8 | 30 | +44.38% |
| Calibration-selected fallback | 0 | 0 | 72 | Undefined edited support |

At matched counts, OLS4 addback chooses the full OLS4 path 240 times
versus 31 in the original arm. Its direct paired full-grid lower gain
versus the original arm is -0.0076 pp (95% source interval
[-0.0283, +0.0045]); full screening gives -0.0266 pp
[-0.0724, +0.0050]. Coupling improves the worst easy change versus
independent full screening but its direct lower gain is -0.0389 pp
[-0.1095, -0.0004]. These are paired bounds recomputed from both sets of
predictions, not differences of their separate bounds versus CV.

**No calibration configuration qualifies (0/9).** The best setting in
each cell passes at most three calibration recordings. The calibrated
policy exactly returns CV, with no edited reference support and no
certification. All intervention arms still trail OLS4 on conservative
utility (about 55.5 to 55.7 pp, normalized by observed CV error mass).
That does not make OLS4 an admissible deployment policy; it passes no
recording view under the same original rule.

The previous hindsight capacity increase was real, but making those
paths available does not let these frozen scores identify safe and useful
edits. I would not tune the relaxed threshold on these held views. The
next model change should learn whether an edit closes the gap to OLS4
while maintaining calibrated CV-relative harm and support at the
recording level, with a matched training control.

`fresh_run`: candidate inference, nine calibration searches, matched
choice freeze and 72-view readout on 24 recordings from nine fitting
sources (45.13 s). `cached_verified`: 18 trained heads, source-excluded
donor paths and exact prior forced/relaxed/gain-match choices. Zero
optimizer updates. The intervals use 3,000 source-group draws after
averaging seeds. These reused fitting sources are exploratory.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_screen_order.py
```

`not_run`: new A readout, new DroneCrowd calibration, any confirmation,
deployment, Stage5C or SMC. The 15/30 DroneCrowd design and the original
2% rule are unchanged. Native 8/12 annotation steps and local pixels are
not verified seconds or physical units. This is not 3D or a foundation model.

[Protocol](protocol.md) | [Metrics](readout.json) | [Choice freeze](freeze.json)
