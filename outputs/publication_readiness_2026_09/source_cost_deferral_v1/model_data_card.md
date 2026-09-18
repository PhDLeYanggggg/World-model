# Training-Only Relative-Cost Deferral: Model and Data Card

## Purpose

This experiment asks whether a causal score can learn when to use a neural
proposal while preserving an exact stationary-baseline action. It is an
optimization diagnostic, not a deployment release or an independent test.
The main 8-observation/12-prediction benchmark is unchanged.

## Model

The inherited proposal has 44,864 parameters. Past geometry, local frame,
coverage and a coverage-only CNN feed a 32-dimensional hidden representation
and a bounded 12-by-2 trajectory head. RGB is zeroed in both new arms. This
experiment is not a new Transformer, JEPA or multimodal-effect comparison.

A 33-parameter linear score head gives 44,897 total parameters. At initialization
its score is zero and the inherited proposal reproduces exactly. At inference
the proposal is used only for score > 0; otherwise the stationary baseline is
returned exactly. Unsupported contexts also return the baseline. The model
receives no future endpoint, future availability feature or target error.
Future errors are loss/evaluation labels only. A bounded offset is not a
physical collision guarantee.

Two fixed training objectives share their initialization and sampled examples:
expected relative action cost, and that cost plus detached signed-gain
supervision and a 0.1 proposal-error term. The second changes two training
components together, so their individual effects cannot be isolated here.
The sigmoid score is not a calibrated probability. The soft expectation is
not a generated stochastic trajectory or the deployed deterministic output.

## Population and Boundaries

The training complement contains 15,430 complete stationary-history queries,
545 recording-scoped agents, 29 recordings and four physical source sites:
coupa, deathCircle, gates and hyang. All source recordings come from the
original SDD train40 assignment. Of these queries, 8,566 have exact-zero future
displacement and 6,864 have nonzero annotation displacement. Annotation change
does not by itself establish an intentional walking start.

Eight supplied past points and twelve future labels use stride 12 raw frames.
The supplied annotations may include interpolation/generated points; this is
not a strict sensor-as-of experiment. Incomplete future labels are not negative
examples. Past normalization and training loss scale are inherited and fixed.

Bookstore's 6,944 queries remain excluded from this fit and are not forecast by
this experiment. They were explored previously and are not a pristine future
confirmation set. Main selection, calibration and confirmation roles remain
closed. Historical Stage37 lineage limitations are not repaired by this run.

## Interpretation and Release

Report candidate trajectories separately from hard-gated actions and soft
expected costs. An all-baseline action has zero gain, not predictive success.
Zero-target CV error is zero, so percentage easy degradation is undefined.
Report absolute annotation-pixel harm without changing the main 2% easy gate.
Training seed ranges do not establish scene-level generalization confidence.

This registration has no threshold search, held scoring, model promotion,
Stage5C execution or SMC. Coordinate and time claims remain annotation pixels,
past-normalized coordinates and raw frames, never verified meters or seconds.
No true-3D, foundation or submission-ready claim is supported.

Only code, configuration, aggregate metrics and original statistical plots are
public. Source media, caches, checkpoints and per-query arrays stay local.
Reproduction requires the registered local assets, not only a code checkout.
See [reproduction](reproducibility.md) and [Chinese operations](operation_zh.md).
