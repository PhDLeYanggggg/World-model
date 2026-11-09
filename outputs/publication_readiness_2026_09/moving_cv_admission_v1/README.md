# Optional CV-Relative Edits With Learned Support and Harm

I separated the choice of candidate from the choice to edit. The head
predicts positive reference and harm masses, their easy shares, and the
probability of wholly missing or positive easy support. It estimates the
unknown-path debit from predicted missing probability and causal peak edit.
The observed 2% rule is unchanged. No A outcomes were used: the nine fitting
sources rotate through separate fit, calibration and held groups. Fit and
calibration see analytic paths only; held paths also include the existing
PCT/EqMotion predictions, each made without fitting that held source.

| Held strategy | Edits across seeds | Original risk passes (/72) | Full-grid lower gain vs. CV (pp) |
|---|---:|---:|---:|
| Forced gain ranking | 27,642 | 0 | -6.791 |
| Strict screen or calibration-selected fallback | 0 | 0 (72 undefined) | 0 (identity) |
| Gain-only at matched query counts | 2,317 | 1 | -0.125 |
| Fixed relaxed diagnostic, independent | 2,317 | 13 | -0.005 |
| Fixed relaxed diagnostic, coupled | 2,317 | 14 | -0.004 |

**None of nine calibrators finds a qualifying configuration.** Even the best trial in
each cell passes only 0 to 3 calibration recordings, fewer than the required
5 to 10. The calibrated and strict arms exactly return CV; zero edits do not
provide a supported reference or an estimable matched comparison.

The fixed relaxed diagnostic changes only the *predicted* risk screen,
not the measured rule. At exactly the same query counts, it improves risk
passes from 1 to 13, but its direct paired lower gain against gain-only is
**-0.353 pp** (95% source interval [-0.523, -0.182]). It is not a safe or
effective operating rule. Coupling changes 37 recipients and adds one risk
pass, but its direct lower gain versus screened independent is -0.013 pp
[-0.038, +0.0001]. The observed pair gain is +0.0013 pixels and its full-grid
lower gain is -0.0255 pixels. Worst signed easy change remains +35.41% for
both screened diagnostics, versus +54.41% for the gain-only match.

## Which part limits the policy?

Using outcomes only after freezing the causal choices, I let a solver
withhold any of the forced nominees. With the original risk and the same
query cap, **60/72** have a verified positive gain over CV alone, but only
**3/72** have one over both CV and OLS4. Keeping the nominated path of every
row rather than pre-pruning to eight expands dual capacity to **6/72**.
Making the original OLS4 path separately available, still leaving CV on
untouched tracks, expands it to **15/72**. All added witnesses were checked
under the original scorer; none is a causal decision or independent result.

The admission head finds safer edits, but ranking first removes useful
options for the stronger reference. The next model should keep OLS4 and
neural alternatives available through screening, then compare a causal
multi-action decoder against gain-only ranking. Tuning a cutoff on the
same pre-pruned nominees is not supported by this diagnosis.

## Reproduction

`fresh_run`: 18 Torch fits, 18,000 updates, three seeds, 24 held recordings
and 9 fitting sources, 82.92 s on local CPU, peak RSS 3.12 GB; fresh
hindsight diagnostics. `cached_verified`: source-excluded donor array hashes
and exact zero-update replay (31.66 s). 25 focused tests pass. Anonymous
replay recreates scalar agent and risk tables, not the training or solver.
The archive SHA-256 is
`b7bdbc25227e2427da953e75200d358e016d76df1787cb1d2c470b7443faa2c2`.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_admission.py
.venv-pytorch/bin/python scripts/run_m3w_cv_admission.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_admission_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_admission_unpruned.py
.venv-pytorch/bin/python scripts/build_m3w_cv_admission_replay.py
.venv-pytorch/bin/python scripts/plot_m3w_cv_admission.py \
  outputs/publication_readiness_2026_09/moving_cv_admission_v1 \
  data/stage_cvpr2027_experiments/moving_cv_admission_v1/admission_capacity.png
```

`not_run`: new A readout, official DroneCrowd calibration/confirmation,
deployment, Stage5C or SMC. All 30 independent groups remain sealed. Native
eight observed and 12 future annotation steps, local image pixels, not
metric geometry, verified seconds, physical safety, true 3D, or foundation.

[Protocol](protocol.md) | [Readout](readout.json) |
[Withholding diagnostic](withholding_capacity.json) |
[Menu expansion](unpruned_capacity.json) |
[Manuscript v126](../evidence_manuscript_v126/README.md)
