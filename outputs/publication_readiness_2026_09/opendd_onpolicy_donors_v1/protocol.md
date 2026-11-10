# Policy-matched donor allocations

Status: prospective development experiment, registered before new fitting.
This is a repair of training/evaluation allocation mismatch, not a new external
confirmation or a claim of a novel imitation-learning algorithm.

## Question

Does training the incumbent-inclusive selector on allocations produced by the
same kind of learned policy used at evaluation reduce destructive exchanges?
The previous donor generators used fixed-priority and random score schedules;
the evaluated incumbent uses a learned recording count head followed by an
exact-count, movement-capped score optimizer. Keep the incumbent-inclusive
features, 48/32 selector, BCE versus BCE+pairwise loss, 400 selector updates,
32 slates per batch, seeds 17/29/43 and the original evaluation roster fixed.

## Nested roles

There are seven exposed OpenDD development locations. For outer location H and
donor D, a donor count producer is fitted only on the other five locations.
Each of those five training locations T receives nominees from a ranker fitted
on the four locations excluding H, D and T. Its own labels supervise the count
head; they never supervise its nominee producer. All target-derived cutoffs
use the respective producer's permitted training locations.

The existing double-excluded rankers nominate actions on D. The new count head
produces D's logits. The original token cap sets the exact edit count and
movement cap, and the original score-only optimizer chooses the incumbent.
Only after that choice are D's labels read to supervise keep and legal
exchanges. No outer-location labels participate in fitting or selection.
Unresolved score optimization is recorded and omitted from donor slates, not
silently replaced with a label-aware or random incumbent.

Unique upstream fits: 35 excluded triples x 3 seeds = 105 rankers, 700 updates
each; 21 excluded pairs x 3 seeds = 63 count heads, 500 updates each. Teacher
menus and receipts are immutable caches. The final selector has 42 fits and
16,800 updates, exactly as in incumbent_slate_v1. The count of donor rows may
change because generation is the treatment; report it rather than claiming
equal training examples. All-location loading does not authorize fitting on
excluded locations.

## Fixed readout

Keep the existing 101 eligible recording-seed views, 780 forecast edits and
all count/movement/token constraints. All 21 outer-location/seed choices must
be frozen before scoring. Report forced and optional variants. Optional argmax
ties keep the incumbent. No gain, confidence or keep-threshold search.

Primary arm: paired optional. To support the repair it must have a positive
lower 95% paired-location bootstrap bound versus (1) incumbent, (2) uniform
exchange on its own switch mask and (3) the previous paired optional selector.
Use 3,000 draws, seed 817, average seeds within location first. Report original,
full-grid and material full-grid pass counts, rescue/destruction, risk-undefined
and risk-violation counts, per-location and per-seed results. Full-grid and
original certificate definitions, empty-support treatment and 2% rule are
unchanged. Do not interpret one favorable contrast as overall success.

All outcomes here are exposed development evidence. Recording-level pooled
history features and the batch allocator are not a streaming deployment
claim. No new external labels, release policy, author Word or disclosure are
changed. Stage5C and SMC stay disabled.

## Runtime and continuation

Native arm64 .venv-pytorch, CPU4, interop1, workers0. Save model, optimizer and
RNG every 100 updates; per-producer teacher caches, process lock, heartbeat and
per-invocation log. Pilot one excluded-pair/seed producer, then use measured
runtime and memory to choose local completion or CREATE. Keep 10 GiB free;
checkpoint at 12-hour boundary if necessary. Resume is the same command, not
a reduced scientific roster. Do not score an incomplete fit.

## Prior work boundary

Ross, Gordon and Bagnell (AISTATS 2011) study prediction-induced distribution
shift and iterative data aggregation. This experiment is one-pass nested
policy-matched donor generation, not DAgger and not its theoretical guarantee.
Source: https://proceedings.mlr.press/v15/ross11a.html (abstract checked).
