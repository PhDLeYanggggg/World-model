# AV2: Separating the Two Release Prerequisites Does Not Transfer

In the [timing-only experiment](../av2_incumbent_timing_v1/README.md),
a retrospective replay rule found 29 qualifying substitutions but
never passed release calibration. I freeze its chosen paths and
change only the order in which scenes can be released.

## Design

Let A be exact eligibility of the original independent path and J
be exact matched qualification of its replacement. J implies A.
One shared fit estimates the direct P(J) and the factored score
P(A) P(J given A). The reference head sees only the original
path and its past context. Conditional loss is computed only for
eligible references. The two controls are the unchanged replay score
and that same score multiplied by estimated P(A). All four select the
same fixed replacement when active; no model chooses a new path.

On each outer fold, donor proposals come from additive and forecast
checkpoints that excluded the proposal's city, the calibration city,
and the held city. The 18 fits reuse the checked inner models and
4,364 past-only donor proposal views (190 eligible references,
75 matched positives). The 8/12 grid, native units, same two
edited actors and peaks, and original both-arm 2% checks are
unchanged. Every distinct nonempty prefix is checked in calibration;
the largest passing one fixes the threshold. Raw replay indices,
paths, scores and its calibration table reproduce before held scoring.

## Results

Of the 1,020 two-slot held views, 935 have a positive replay proposal.
Of those, 57 have an eligible original path and 29 have a matched
replacement. The latter is a **3.10%** positive rate on proposals,
not a release rate.

| Score | Accepted calibration views | Held pass / release |
|---|---:|---:|
| Replay | 0 | No release |
| Replay times P(A) | 0 | No release |
| Direct P(J) | 1 | No release |
| Factored P(A) P(J given A) | 5 | 0 / 1 |

The factored score's pooled Brier error is 0.03314 versus
0.03473 for direct. The paired city-balanced difference is
**-0.00078 [-0.00717, +0.00381]**, after averaging seeds within
scenario and resampling cities 3,000 times. Its average precision
is 0.0372 versus the 0.0310 positive rate. The conditional head
on the 57 eligible views has AP 0.4697 versus prevalence 0.5088
and Brier error 0.4567. Detecting some eligible reference paths
does not make the improvement conditional on eligibility predictable.

The one factored release (Washington DC, seed 43) improves
conservative trajectory gain by 1.98 pp against its incumbent.
But that incumbent is ineligible. The new path loses to
OLS4 by 2.56 pp; its positive easy-harm ratio is 361.78% and its
signed easy degradation is 27.96%. A positive comparison to an
ineligible incumbent does not pass the original matched evaluation.

I would keep the candidate-supply result in the supplement but not
make a positive AV2 deployment claim. With the same 29 matched
proposals, a new release score recovers no qualifying held case.
The next priority is a broader source-separated comparison with
enough eligible references and genuinely independent confirmation,
not further threshold search on these exposed cities.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_replay_hurdle.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_replay_hurdle.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads, one process. Fit/freeze took
34.26 s at 0.68 GB maximum RSS, held scoring 5.13 s. Eighteen
fresh fits, 74 focused tests. Atomic checkpoints every 40 updates
include optimizer and sampler state; run the same fit command to
resume. No CREATE submission, raw-data acquisition, reserved outcomes,
manuscript edit or deployment change.

### Separate supplement copy

The builder appends one self-contained English page to the verified
54-page budget supplement. The original PDF and manuscript are
unchanged. Existing page streams, 275 named destinations and the
evidence guide links are checked; the new outline entry opens page 55.
The original 54 pages render identically at 72 DPI; page 55 was
rendered and inspected at 110 DPI. A separate build is byte-identical.
The PDF is local, not in Git.

```sh
# Use the bundled Python with pypdf, reportlab and numpy.
python scripts/build_m3w_av2_calibration_addendum.py \
    data/stage_cvpr2027_experiments/submission_admin/supplement_with_av2_calibration.pdf
```

PDF SHA-256:
`57b196563849020d1d608f0b992cf022c6e9effef06c90538cb68b40e1a0d2c6`.

[Registration](registration.json) · [Freeze](freeze.json) · [Results](summary.json)
