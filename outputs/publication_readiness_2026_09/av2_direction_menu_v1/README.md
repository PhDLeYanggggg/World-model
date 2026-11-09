# AV2: Direction and Timing at the Same Edit Budget

The [timing study](../av2_incumbent_timing_v1/README.md) found useful
substitutions without changing recipients or peak displacement. I
now compare changing the direction of those same two edits against
changing their timing. This tests candidate supply before refitting
another selector.

## Comparison

I keep the same incumbent, two edited actors, and each actor's exact
CV-relative peak displacement. The timing bank keeps the previous
early and late profiles. The direction bank instead rotates each
edited actor's whole CV-relative offset sequence by -15, 0 or +15
degrees. Rotation preserves the original offset magnitude at every
step. Each standalone bank has eight alternatives plus the incumbent;
the union has sixteen alternatives. I do not combine rotation and
timing in the same path or claim a same-size comparison for the union.

The existing eight-observation replay score ranks all three banks.
Each fold uses the original separate calibration city and strict
release rule. Held candidates, choices and thresholds are sealed
before held scoring. No model is refit. Hindsight coverage requires
the unchanged incumbent and replacement to both beat CV and OLS4,
pass the original three 2% checks with support, and show a pair gain
without worsening trajectory error. Standalone eligibility removes
the incumbent and relative improvement requirements, not the
motion floors or risk checks.

## Results

| Candidate bank | Alternatives | Standalone eligible | Qualifying alternative to incumbent | Qualifying top replay choice | Held releases |
|---|---:|---:|---:|---:|---:|
| Timing | 8,160 | 70 | 51 | 29 | 0 |
| Direction | 8,160 | 65 | 50 | 28 | 0 |
| Union | 16,320 | 71 | 62 | 31 | 0 |

The 1,020 views are 340 development scenarios across six cities,
repeated for three seeds. Direction alone does not improve the
equal-size comparison: the city-balanced change in matched coverage
is **-0.320 pp [-1.232, +0.361]**. Adding direction to timing gains
**+0.787 pp [+0.251, +1.468]**, or eleven additional views overall.
The 3,000 paired bootstrap draws resample cities after averaging
seeds within each scenario. These are outcome-aware opportunities,
not released forecasts or gains in trajectory-error units.

The union covers 62 of the 63 eligible fixed references. Among
465 fully labeled views, the three banks cover 33, 38 and 43
matched opportunities respectively, but only 43 original references
qualify. No menu passes release calibration. Even the union improves
replay retrieval by only two views over timing. The complete-label
slice shows that missing labels alone cannot explain the scarcity
of usable original forecasts.

The local bank is now close to exhausting the available fixed
references. The next useful experiment is a stronger initial
forecast under the same actor and displacement budget, with
standalone qualification measured before any paired selection.
A larger local menu or another release classifier on the old
incumbents cannot remove this ceiling. These are development results;
no reserved outcomes, deployment or manuscript were changed.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_direction_menu.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_direction_menu.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_direction_menu.py replay
.venv-pytorch/bin/python scripts/package_m3w_av2_direction_menu.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads, one process. Freeze took 18.90 s
at 0.64 GB maximum RSS, and scoring took 9.64 s. The focused suite
passed 76 tests. Replay recomputes the summary from fold receipts.
The local anonymous `submission_admin/av2_direction_menu.zip` was
built twice with identical bytes. It uses only anonymous scalar
counts and the original bootstrap draws. Run `python -I -B -S verify.py`
after extraction; it recomputes counts and six paired intervals.
It does not retrain or read raw labels or trajectory arrays.

ZIP SHA-256:
`96f13077435aadafce6687efd4d5073709f54a6e5d0ac17cc3db4fa965755ee0`.

[Registration](registration.json) · [Frozen choices](freeze.json) ·
[Results](summary.json)
