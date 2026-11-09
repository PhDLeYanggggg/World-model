# Does the direction of relative motion help the pair score?

The [direct pair-effect head](../opendd_direct_pair_effect_v1/README.md)
rarely predicted its held target more accurately than zero. I tested one
possible cause: the pair features knew the sizes of displacements but not
whether two agents were approaching each other, or whether their edits
pointed with or against their relative motion.

## Matched comparison

- The two heads have the same architecture, initial weights, sampled pairs,
  targets, optimizer, batches and seeds. The control receives zeros in the
  13 new channels. Both heads keep the same frozen unary score and feature
  extractor. The signed channels use past radial approach, projections of
  candidate edits, and pair-effect proxies from CV and past-only linear fits.
  They are invariant to translation, rotation and exchanging the two agents.
- Both arms use the exact previous training examples: the original inputs,
  geometric summaries and target arrays match bit for bit on every supported
  source. The target remains the additional relative-error reduction of two
  edits beyond their single-edit effects on co-observed steps. There are no
  future labels in the additional channels.
- Seven held OpenDD development locations, three seeds and 42 new heads.
  The two arms train side by side for 6,300 optimizer steps, with each held
  location excluded. Two locations have no legal pair probe and remain in
  policy scoring. Forecasts are frozen before scoring the held policies.
- Every arm searches the same exact path menu, per-query edit count and
  peak displacements. The 1,556 singleton queries must be identical.
  The original 2% risk rule and missing-step debit do not change.

## Results

| Arm | Edits | Risk passes / 606 | Risk and both references / 606 |
|---|---:|---:|---:|
| Additive | 3,690 | 39 | 32 |
| Magnitude only | 3,690 | 40 | 33 |
| Magnitude plus signed geometry | 3,690 | 40 | 33 |

Signed geometry reduces pair-target RMSE versus magnitude alone on **11 of
15** supported held location-seed probes. It beats predicting zero on only
**3 of 15** (magnitude alone: 2 of 15). It changes 108 of 1,067 double-edit
choices and none of the 1,556 singleton choices.

| Signed head comparison; positive is better | Site mean | 95% location bootstrap |
|---|---:|---:|
| Conservative utility vs. magnitude (pp of CV error) | +0.00008 | [-0.00623, +0.00602] |
| Conservative utility vs. additive (pp of CV error) | +0.00028 | [-0.00794, +0.00747] |
| Observed pair-relative error reduction vs. magnitude (native units) | +0.00002770 | [+0.00000203, +0.00006017] |

On `rdb6`, relative geometry improves while conservative trajectory utility
falls by 0.01293 pp. The 606 views are 202 recordings repeated at three
seeds. Intervals draw 3,000 bootstrap samples from six estimable locations;
the seventh has no edits and no matched utility estimate. These are
descriptive development results, not a physical safety measure or a new
external confirmation.

**Decision:** Direction is informative enough to change pair predictions and
slightly reduce the observed geometry error. It does not produce a useful
advantage under the matched risk and accuracy criteria. Do not scale this
head or claim a deployment benefit from the geometry result alone.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_signed_pair_geometry.py freeze
.venv-pytorch/bin/python scripts/run_m3w_signed_pair_geometry.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_signed_pair_geometry.py tests/test_m3w_pair_effect.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_slate_imitation.py
```

`fresh_run`: native arm64 CPU fitting, matched inference and scoring.
`cached_verified`: source cohorts, unary checkpoints and intervention budgets.
Four compute threads, zero worker processes, a PID heartbeat and atomic
checkpoints every 50 steps. Seventeen focused tests passed. No reserved
outcomes were opened, and deployment, manuscript and submission are unchanged.
