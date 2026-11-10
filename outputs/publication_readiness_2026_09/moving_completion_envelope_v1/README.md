# When a displacement limit rules out beating OLS4

I revisited the same 24 exposed MovingDroneCrowd++ recordings (nine source
groups, three seeds each) to separate the displacement limit from uncertainty
about the unobserved future. The budget, eight observed positions, twelve
prediction steps and query weights are unchanged. This is a hindsight upper
bound on accuracy, not a model, a risk pass, or an external result.

| Accuracy criterion | Fixed query budget: views excluded | Pooled recording budget: views excluded | Pooled median ceiling (pp) |
| --- | ---: | ---: | ---: |
| Conservative missing-step score | 67/72 | 67/72 | -27.042 |
| Observed steps alone | 43/72 | 36/72 | -0.159 |
| Most favorable completion | 9/72 | 9/72 | +31.167 |

The nine favorable-completion exclusions are three recordings, each excluded
in all seeds. No entire source is excluded this way. In 58 other views the
conservative score rules out a gain, but the favorable ceiling does not. A
nonnegative ceiling is not a win: it drops the edit-count and risk constraints
and may use implausible missing positions. The 72 views are repeated runs,
not independent recordings.

## Derivation

Let `c`, `o`, `p` be CV, OLS4 and a replacement at one step, with
`||p-c|| <= r`; each agent has a single peak allowance `r` shared across its
steps. On a labeled step, the best replacement error is
`max(||c-y||-r, 0)`. Relative to CV, this improves the observed gain by
`min(r, ||c-y||)`. On a missing step, the largest possible OLS4-minus-replacement
error is `||c-o||+r`: choose `p` away from `o` and take the missing label to
be `p`. The triangle inequality gives the same upper bound without any
assumption on the missing label. For the conservative score, the corresponding
step is `-max(||c-o||-r, 0)`, since the best replacement moves toward OLS4.

The favorable objective adds an affine term `missing_steps * r` per agent
to the observed gain, plus the fixed missing CV-to-OLS4 distance. The sum is
piecewise linear and concave in the allowances. With the recorded peak budget,
sort the marginal slopes after query weighting. Slopes below the strongest
unbounded missing-step slope cannot improve the allocation. A fractional
allocation and its dual give matching values. Allowing independent completion
of each missing waypoint only enlarges the feasible set, so a negative ceiling
also rules out positive gain when completions must agree across overlapping
forecasts. Scores are percentage points on the observed CV denominator, not
raw ADE or metric distances.

## Reproduce

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_moving_completion_envelope.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_moving_completion_envelope.py tests/test_m3w_moving_peak_certificate.py
```

The calculation reads the frozen development forecasts and labels, checks the
original certificate and break-even results, and writes a [light receipt](receipt.json).
Tests compare the grouped computation with an independent actor-level linear
program and check complete and partially labeled examples. No confirmation
labels, new forecasts, or changes to the original 2% rule are needed.
