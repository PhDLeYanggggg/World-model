# Data Card: Frozen SOURCE Estimator Comparison

## Source And Task

The experiment reuses the frozen EuropeanSquares SOURCE development lineage.
Coordinates are oblique webcam detector-track box midpoints in pixels, not
SDD human-annotated top-down ground truth. The task observes eight steps and
requests twelve future steps at stride twelve raw frames. Effective seconds
and metric calibration are not established by this experiment.

The fixed roster contains twelve SOURCE localities. Prior verified records
identify 74 unique fitting recordings and 31 inner-held recordings. Contexts
and controller seeds repeat their views; these repetitions must not be counted
as new independent scenes or independent trajectory training pipelines. Final
fresh readout receipts report exact view counts and known/unknown support.

## Roles And Label Support

Only original whole-record fitting packets train the estimator. Normalization
uses fitting physical actions; query draws reproduce the frozen neural cost
head's source-balanced sampling. Unknown future costs are not sampled as zero
labels. Actual forecasts, SOURCE packets and static mappings are reused with
hash/schema checks; estimator fitting and its inference readout are fresh.

Native ADE uses an available nonempty future mask. No observed future outcome
means unknown, not perfect prediction. FDE can have different support. Future
targets and masks are available only to training losses/evaluation, never to
the causal inference function. Easy/hard cuts are fitting-defined evaluation
events, not oracle input features.

SOURCE inner-held has already been exposed in development. This new readout
does not restore independent-test or whole-pipeline OOF status. Independent
selection, calibration, confirmation and DroneCrowd are not opened.

## Features And Context

The same 380 causal motion/neighbor/rollout descriptors and 18 frozen static
image-region descriptors are used for the two estimator families. Region
polygons are not verified walkable SDFs or homographies. The causal envelope
comes from forecast disagreement, not error against a future endpoint.
No test-endpoint goal construction, central velocity, test normalization,
recording identifier or eventual-label mask is supplied to inference.

A current-bank expanded fitting matrix contains one row per sampled real
query/action, with duplicate-query multiplicity represented as weight. It is
not a full materialization of trajectory episodes. Its sizes and hashes appear
in each fitting receipt; it is not committed to the public repository.

## Statistics And Distribution Limits

The primary uncertainty calculation resamples SOURCE locality means after
averaging repeated context/head-seed views. Known hard/easy pooled-error ratios
are supplementary. Empty required views stay present and can make a macro
not_estimable. Unknown costs remain in completion bounds and full risk checks.

This design does not establish transfer to unseen datasets, metric physical
safety, fully observed twelve-step accuracy, or broad world-model competence.
No raw videos, images, third-party trajectories, working matrices or weights
are published with these aggregate reports.
