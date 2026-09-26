# Auxiliary Gradient and AdamW Intervention Diagnostic

Registered before fresh diagnostic readout. Parent: the sealed strong-base
auxiliary study at f1d011aae570608f28a714ecafaee27c0cb0b10a. This is a fitting-only
mechanistic experiment, not another independent training or policy benchmark.

## Question

Does the auxiliary task locally interfere with the four-cost estimator, and
does removing its conflicting shared component improve actual AdamW updates?
Negative cosine alone is not a causal explanation of the previous failure.

## Fixed Design

- All 144 original views, three seeds (17/29/43), full/motion families and
  three frozen training arms. All 432 step-2000 models and optimizer states.
- Eight deterministic, site-balanced batches per state, 256 rows each.
  Draw 256 disjoint-row fitting probes per fitting locality. The original
  model has already trained on these populations: these are NOT held-out
  validation, independence or generalization checks. Overlapping windows
  and repeated batches are not independent samples.
- Use original fitting-only normalization, nested OOF four-cost targets,
  row-locality-excluded auxiliary labels, RMS scales, architecture, optimizer
  moments, learning rate .0003, weight decay .0001 and global clip 5.
- At each SAME model/optimizer state compare five isolated one-step updates:
  cost only, true auxiliary, within-locality shuffled auxiliary, projected
  true auxiliary and projected shuffled auxiliary. Each branch resets to
  the frozen state. No sequential accumulation, no checkpoint replacement.
- Report shared and full task gradient cosine, norms and norm ratios for
  cost4 versus BCE, and positive-envelope easy-harm versus BCE. Initial
  model shared gradients are also checked. Zero norm means not_estimable,
  never evidence of agreement. Empty event support is reported explicitly.
- Projection removes only the negative shared auxiliary component along
  the four-cost gradient. The primary gradient and auxiliary-only head
  gradients are unchanged. It is not symmetric PCGrad and does not guarantee
  improvement of the easy-harm component or safety under AdamW/global clipping.
- Fresh measurements: 17,280 virtual AdamW steps; 3,456 final-state batch
  diagnostics and 144 initial-state checks. No new fully trained head.
- Measure actual before/after four-cost, all-row easy-harm and positive-
  envelope easy-harm normalized MSE on fitting probes. Future outcomes are
  loss labels only; no target/event/oracle becomes an inference feature.

## Aggregation and Decision

Average eight repeats, contexts and seeds within fitting locality, then use
3000 paired locality resamples separately for each of six producer/controller
assignments, feature family and frozen training arm. Four source localities
remain shared across contexts and models. Intervals describe finite fitting
probe variation only, not an independent sample or confirmatory test.

Primary repair screen uses full-input, cap_aux frozen states. A full-training
projection repair is warranted for separate preregistration only if projected
true versus true auxiliary has positive point changes in BOTH cost4 and
positive-envelope easy-harm in at least five of six assignments, with no
strictly negative interval; projected true versus projected shuffled must
also have positive points in at least five assignments for both metrics.
This is an exploratory allocation rule, not a scientific success gate.
All arms, metrics, assignments and null/negative results are retained.
No checkpoint, threshold or model is selected on outer outcomes. If this
screen fails, do not call gradient projection a supported repair; examine
trajectory/target transport with a new preregistration instead.

## Scope and Prior Work

[Du et al.](https://arxiv.org/abs/1812.02224) motivate auxiliary-gradient
similarity as a local heuristic; it is not a positive-transfer guarantee.
[Yu et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
study task-gradient projection and multiple interacting sources of conflict.
This one-sided diagnostic adapts known ideas and is not claimed as a novel
method. Actual AdamW effects, not an SGD geometric argument, decide the screen.

Source development only. Independent selection, reserved calibration and
confirmation stay unopened. Obs8/pred12 native annotation steps, detector
pixels. No metric, seconds, physical-safety, human-gold, true3D or foundation
claims. No deployment change, Stage5C execution or SMC. Local native arm64,
CPU4/interop1/workers0. Atomic per-view receipts allow resume. Preserve 10GiB
free disk; do not modify other CREATE jobs or unrelated staged work.
