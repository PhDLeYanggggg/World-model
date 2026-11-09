# Can a past-only policy recover full-path gains?

Not on this exposed camera. Full-size candidates support
future-informed plans that beat CV and OLS4 under the original
2% risk rule. A donor-trained value head does not recover them.
The default forecast remains CV.

## Causal selection and matched controls

I trained 42 spline producers (seven held OpenDD sites, two
training arms, three seeds) and three value heads. The head sees
donor labels and predictions from models that did not train on
the predicted site. The camera uses the all-donor spline
checkpoints already frozen in the previous experiment. No camera
label, test threshold or future visibility enters selection.

On the Convenience Store pilot, 57 of 58 online histories have a
stable reference match and a complete future. The seed views share
one camera. I keep the unmatched row in the risk and utility
calculation. There is no cross-camera confidence interval.

| Seed | Full-size hindsight lower gain vs OLS4 (pp) | Raw policy edits | Equal-peak edits | Equal-peak joint vs OLS4 (pp) | Equal-peak joint vs independent (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | +15.42 | 0 | 23 | -17.42 | 0.00 |
| 29 | +14.57 | 0 | 23 | -17.07 | 0.00 |
| 43 | +13.52 | 0 | 23 | -16.76 | 0.00 |

These are conservative paired points on the CV denominator.
At full size, the model predicts positive gain for 53--59 out of
232 available single actions, but no action has nonpositive
predicted easy excess. No pair clears the predicted checks, even
before the pair penalty. All three raw policies abstain and
therefore supply no positive selected-risk reference.

The equal-peak study fixes count and displacement per query,
within each menu. For the expanded menu, the joint, independent and
no-pair variants give exactly the same output paths in each seed.
The joint contrast against one common family ranges from -0.68 to
+1.70 pp. The matched gain-only comparator beats the expanded
risk-aware rule by 0.64 to 0.99 pp. In the analytic menu the learned
joint rule beats a simple past-prefix ranker by 3.92 to 4.17 pp,
but neither comparison establishes superiority over OLS4.
No expanded equal-peak rule passes both checks. Even hindsight at
that radius has a negative risk-constrained upper bound against
OLS4 (-9.40 to -8.78 pp). Thus the equal-peak result tests
selection at that budget, not at the full-size bank's capacity.

## Repair: estimate signed risk directly

I kept the forecasts and gain estimates fixed and replaced the
two predicted positive-harm-minus-reference differences with
direct signed excess estimates. The three new heads trained on
the same donor banks. No risk threshold or evaluation check changed.

| Seed | Raw edits | Easy positive-harm upper bound | Conservative gain vs OLS4 (pp) |
| --- | ---: | ---: | ---: |
| 17 | 9 | 15.66% | -22.93 |
| 29 | 16 | 32.97% | -22.06 |
| 43 | 10 | 87.52% | -23.65 |

The repair removes complete abstention but produces **0/3**
passes. Gain against CV is positive in all three views, which
does not compensate for the easy harm or the OLS4 deficit.
The next useful model change is to train selection against
verified joint action plans on source-excluded donor folds,
with an analytic-bank and independent-selection control.
A real camera donor split with enough matched histories is
still needed to calibrate harm. Do not open the sealed
confirmation data to choose between these variants.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_full_path_selection.py freeze
.venv-pytorch/bin/python scripts/run_m3w_full_path_selection.py score
.venv-pytorch/bin/python scripts/audit_m3w_full_path_selection.py
.venv-pytorch/bin/python scripts/run_m3w_full_path_signed_repair.py freeze
.venv-pytorch/bin/python scripts/run_m3w_full_path_signed_repair.py score
```

Native arm64 Torch, four threads, zero workers, 100-step atomic
checkpoints. The first run stopped at the value-loss adapter after
14 producer fits; the second registration links and reuses those
completed predictors. No model or data change was made to those
fits. In total 48 fits and 38,400 updates, about two minutes for
the main fits including the restart, plus the signed-risk fits.
Nine focused tests passed. Checkpoint replay matched all frozen
path arrays and scores; the capacity calculation also replayed.
No CREATE submission, external WEPDTOF rescore, DroneCrowd
confirmation, metric conversion, Stage5C, SMC or deployment change.

[Protocol](protocol.md) | [Repaired registration](registration_v2.json) |
[Readout](readout.json) | [Capacity and risk diagnosis](failure_analysis.json) |
[Signed-risk readout](signed_readout.json) |
[Working manuscript v163](../evidence_manuscript_v163/README.md)
