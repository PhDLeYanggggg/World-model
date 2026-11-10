# An explicit keep option does not repair exchange selection

I trained two incumbent-aware rankers to test whether earlier failures came
from forcing a change. Both models score the unchanged allocation alongside
legal exchanges. One uses pointwise binary loss; the other adds paired
comparisons against the incumbent. Forced and optional decisions are read from
the same fitted network, so their contrast does not confound model fitting.

This is exposed OpenDD development: seven locations, 202 recordings and three
seeds. The 606 recording-seed views are not 606 independent recordings.
All arms retain 780 trajectory edits; an allocation exchange moves an existing
edit and does not add one. Only the prior ranker's 101 eligible views can change.

## Results

| Policy | Allocation exchanges | Original dual passes / 606 | Full-grid dual passes | Material full-grid passes |
| --- | ---: | ---: | ---: | ---: |
| Unchanged incumbent | 0 | 60 | 55 | 20 |
| Archived aligned ranker | 101 | 47 | 44 | 17 |
| Pointwise, forced | 101 | 51 | 46 | 18 |
| Pointwise, optional | 95 | 51 | 47 | 19 |
| Paired, forced | 101 | 50 | 45 | 16 |
| Paired, optional | 94 | 52 | 47 | 17 |
| Uniform exchange on pointwise optional's switch mask | 95 | 52.16 | 47.94 | 16.33 |
| Uniform exchange on paired optional's switch mask | 94 | 53.48 | 49.00 | 15.74 |
| Uniform including keep on all 101 eligible views | expectation | 52.74 | 48.24 | 15.97 |
| Hindsight full-grid oracle including keep | diagnostic | 102 | 100 | 36 |

Dual passes require risk support, the original 2% checks and positive utility
against both CV and OLS4. Full-grid also checks intermediate forecast steps.
Material additionally requires more than 0.10 percentage points against both
references. Oracle choices maximize full-grid status only; their original and
material counts are accompanying outcomes, not separate oracle maxima.
Fractional random counts are exact expectations, not trained results.

Paired optional versus unchanged has an equal-location full-grid rate change
of **-1.215 pp**, 95% interval **[-3.120, 0.000]**. Against uniform on its own
switch mask it is **-0.549 pp [-1.481, -0.026]**. Neither primary comparison
supports the method. Paired optional versus pointwise optional is exactly
zero at the location-aggregate level, despite choosing different actions in
40 eligible views. These intervals use 3,000 paired location resamples after
seed averaging; they are descriptive development intervals, not independent
confirmation after the many preceding experiments.

Adding keep to paired inference recovers two full-grid passes, from 45 to 47.
Its equal-location change is +0.075 pp [0.000, +0.225], not evidence of a stable
advantage. Pointwise optional recovers one pass. Neither beats the unchanged
incumbent in any of the three seed totals:

| Seed | Incumbent full-grid | Pointwise optional | Paired optional |
| --- | ---: | ---: | ---: |
| 17 | 19 | 15 | 17 |
| 29 | 17 | 14 | 13 |
| 43 | 19 | 18 | 17 |

Seven net lost full-grid passes are in rdb1 (43 to 36), and one in rdb3
(3 to 2). The remaining five locations tie the incumbent. Two of those
locations have no full-grid passes in any compared arm; a tie there is not
evidence of useful or safe release.

## What failed

The diagnostic replay verifies every frozen choice against its checkpoint.
It does not refit models or choose a new margin.

- **The keep option is undervalued.** On the 101 eligible views, the paired
  head's mean sigmoid score for the incumbent is 5.36%, while 26.73% actually
  pass the original training-target rule. Its chosen action averages 38.00%
  predicted versus 18.81% observed. Chosen-action ECE is 0.352. These are
  uncalibrated score diagnostics, not deployable probabilities.
- **Most actions still exchange.** Paired optional changes 94/101 allocations;
  it rescues five full-grid passes but destroys thirteen. There is a passing
  action in 69 eligible views, but it misses 53 of them. Merely exposing a
  keep option does not teach safe admission.
- **Pairing is not the missing ingredient.** The paired and pointwise optional
  arms both end at 47 full-grid passes and have identical location totals.
  Pairing changes actions and scores without improving this endpoint.
- **Donor supervision differs from scored incumbents.** The cached training
  builder constructs fixed-priority and random schedules. Evaluation uses
  the archived learned allocation. Only 473/4,507 donor incumbents pass
  (10.49%), versus 27/101 eligible scored incumbents (26.73%). This is evidence
  of a supervision-distribution mismatch, not proof that it alone causes the
  failure. Donor totals repeat across folds and seeds and have different
  sampling weights from evaluation.
- **Many paired labels provide no ranking contrast.** Of 4,507 donor slates,
  2,678 are all negative, 19 all positive and 1,810 heterogeneous. The paired
  term can distinguish the incumbent from an option in the heterogeneous
  group only. There are 1,356 potential-rescue and 454 potential-harm slates.
- **Support remains a separate obstacle.** The unchanged arm has 397
  undefined-risk views; paired optional has 398. Keeping an allocation does
  not certify unsupported risk. The unchanged allocation is not plain CV.

The shared features do include allocation-weighted context. It would be wrong
to describe this failure as a complete absence of allocation information.
Whether more detailed selected-agent risk information helps is unproven.

## Next discriminating experiment

The next priority is to align donor incumbent generation with the scored
policy, using producers that exclude both donor and outer locations. Compare
that one change against these fixed/random donor schedules before adding
another model or changing thresholds. Audit producer nesting first; the same
held location must not train the donor actor, its teacher or its preprocessing.
Keep forecast banks, count budgets, risk definitions and external policies
fixed. No new independent cohort should be spent on this failed selector.

## Execution and provenance

`fresh_run`: 42 real Torch fits, 16,800 updates total; final recovery completed
13,600 remaining updates in 57.19 seconds on native arm64 CPU with four Torch
threads, one interop thread and zero workers. The initial pilot and interrupted
run had already completed 3,200 updates. An empty-donor edge case interrupted
training, and a registration-object comparison error interrupted the first
recovery attempt before fitting. Both repairs are recorded in
[runtime_amendment.json](runtime_amendment.json); no targets had been scored.
The four completed cells and their checkpoints were retained unchanged.

A second complete `fit` invocation replayed all 21 cells in 13.14 seconds with
zero updates and identical choices. The freeze was committed as `b7c6a6c39`
before scoring; scoring took 9.52 seconds. Checkpoints retain model, optimizer
and sampler state every 100 updates. All 42 saved training-loss traces are
finite. This is small-head compute on existing donor features, not a new
end-to-end world-model training run.

`cached_verified`: donor features, labels, split roles, forecast menus and
archived unchanged/aligned controls. `not_run`: new external confirmation,
deployment, Stage5C and SMC. CREATE was checked read-only; the four recorded
jobs were COMPLETED with exit 0:0. No remote work was submitted or restarted.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_incumbent_slate.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_incumbent_slate.py score
.venv-pytorch/bin/python scripts/audit_m3w_incumbent_slate.py
```

Repeat `fit` to resume or verify the same choices. `score` verifies artifact
identities and never updates a model. These artifacts are immutable; revised
scientific designs need a new experiment directory. Aggregate JSON uses sparse
count keys: absent rescue/destroy keys represent zero, not missing evidence.
The frozen summary assumes its archived controls have material-positive views;
it is not a general-purpose scorer for a completely empty benchmark.

The existing recording-wide feature pooling makes this a retrospective batched
allocation experiment, not stream-time deployment. No held future labels enter
features. The result does not establish neural dynamics superiority, metric
world modeling, or submission readiness. The author's Word and manuscript
disclosure remain unchanged.

Verification: 24 focused tests passed across the incumbent, listwise, risk,
attrition, full-grid, context and original ranker modules. This is not a claim
that the entire historical test suite was rerun.

[Protocol](protocol.md) | [Freeze](head_freeze.json) | [Readout](readout.json) |
[Diagnostic](failure_diagnostic.json)
