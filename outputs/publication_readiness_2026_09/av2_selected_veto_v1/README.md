# AV2: Training on Selected Actions Helps Overall Harm, Not Release

The [coverage study](../av2_harm_quantile_v1/README.md) found
that the same harm estimate covers fewer outcomes when it
also selects the action. I tested a separate veto trained
on what the ranker actually chooses, rather than on a
uniform sample of legal options.

## Matched Test

For every outer donor city I rebuilt the margin and upper
harm ranker on the other three donor cities. The EqMotion
checkpoint and easy cutoff also exclude the recipient, outer
calibration, and outer held cities. The inner ranker uses
that same producer for its own three-city training inputs
(in-sample for the producer there), then ranks the recipient
city's out-of-city forecasts. It does not see any recipient
outcomes before making those choices.

The two vetoes use identical networks, initialization, donor
scenes, training steps, and loss. In each scene one sees a
uniformly sampled legal assignment, the other the one chosen
by the inner ranker. There are 245-300 donor queries per
outer fit and 4,776 dependent selected training views over
the 18 folds. Neither veto is used to select another path.
The existing outer upper-harm forecasts, gain/support
heads, two-slot budget, and scene roster replay exactly.

Both vetoes rank the frozen choices for possible release.
Every nonempty prefix uses the existing exact reference
qualification with the two causal motion floors and original
2% harm and easy checks. This is a development experiment
on geographic transfer, not a statistical risk certificate.
It is related to [selection-conditional prediction](https://doi.org/10.1093/jrsssb/qkaf016)
and [Learn then Test](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk),
but does not use their procedures or claim their guarantees.

## Evidence

| Score on the same frozen choices | Overall harm coverage | Easy positive harm coverage | Signed easy coverage | Passing calibration prefixes / 1,020 | Held eligible / released |
|---|---:|---:|---:|---:|---:|
| Reuse optimizing estimate | 70.00% | 70.39% | 70.29% | 0 | No release |
| Independent, uniform training | 71.27% | 75.49% | 70.78% | 2 | 0 / 4 |
| Independent, selected training (primary) | 74.90% | 74.71% | 71.76% | 3 | 0 / 3 |

For selected versus uniform training, paired city-balanced
coverage changes are **+4.81 pp [+0.26, +7.98]** for overall
harm, -0.29 pp [-3.74, +3.75] for easy positive harm, and
+1.32 pp [-3.95, +5.74] for signed easy harm. Compared to
reusing the optimizing estimate, the overall change is
+5.10 pp [+0.55, +9.42]. These are 3,000 paired city draws
after averaging seeds within each scenario; the table
pools 1,020 views from 340 distinct two-slot scenarios.
The rest of the 896 held development scenarios are retained
in the full deployment gain calculation.

On the frozen choices, false-safe signs for overall harm
fall from 36 (reuse) and 33 (uniform) to 15 (selected).
The corresponding easy positive counts are 50, 33 and 36;
the signed easy counts are 51, 48 and 49. The upper estimates
therefore improve the overall term without resolving either
easy term. The underlying forced 2,952 edits are unchanged
and qualify in 75 views for every arm.

All three selected-veto held interventions fail original
risk. The uniform veto fails original risk on three of
four and the remaining one still fails exact qualification.
The selected arm's conservative trajectory release gain
versus CV is -0.00314 pp [-0.00942, 0]. The absence of a
release in the self arm is not counted as qualification.
I keep the existing fallback and do not promote either veto.

This result narrows the next question: an independent
estimate trained on selected actions can recover some
overall coverage, but the easy reference and easy harm
need a separate decision target. More overall-harm
training or threshold adjustment on these same outcomes
is not the next experiment. The earlier
[CREATE selected-set calibration](../european_selected_set_calibration_v1/conclusions.md)
also found no useful release improvement under a different
development protocol; it is not pooled with this AV2 result.

## Figure and Anonymous Replay

The local `submission_admin/av2_selection_coverage.svg`
and PNG show the earlier matched post-selection coverage
loss. The dashed line is the nominal 0.8 training quantile,
not the risk threshold. The figure is generated from
frozen results and has not been inserted into the manuscript.

`submission_admin/av2_selection_evidence.zip` contains the
anonymous paired counts and a standard-library verifier.
It enumerates all 46,656 bootstrap draws of six city
clusters and independently checks all three earlier
coverage drops. Exact conditional intervals are
[-12.09, -0.89], [-10.05, -2.23], and [-15.17, -6.67] pp.
The ZIP replays outside the repository, and two builds
match byte-for-byte. SHA-256:
`23292c5cb772e5ce91274fca2c9ba55ed63bd1aa4c06f7c9f58549c7c714a964`.
The plot and ZIP stay local; the generator and verifier
are tracked. The replay verifies scalar arithmetic, not
model training or independent confirmation.

## Reproduce

Requires the [upper-harm assets](../av2_harm_quantile_v1/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_selected_veto.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_selected_veto.py score
.venv-pytorch/bin/python scripts/plot_m3w_av2_selection_coverage.py
.venv-pytorch/bin/python scripts/package_m3w_av2_selection_evidence.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64 PyTorch, four threads, zero data workers.
72 inner ranker fits (each with a margin and upper-harm
component) and 36 independent veto fits completed in
405.63 s with 783 MB peak RSS; scoring took 4.90 s.
Atomic checkpoints every 40 updates retain optimizer and
sampler states. Completed receipt replay took 7.54 s.
56 focused tests passed. No new CREATE job, reserved
confirmation access, manuscript edit, Stage5C, or SMC.

[Registration](registration.json) · [Freeze](freeze.json) · [Results](summary.json)
