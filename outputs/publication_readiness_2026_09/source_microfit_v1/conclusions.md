# Microfit Diagnostic: Learnable Training Rows, No Generalization Claim

## Material Passport

`fresh_run`:12 real Torch fits and24,000 updates, three seeds, two decoder laws
and two selected training microcohorts. Summed fitting time399.41seconds;
full-run log span6.685minutes. The100-update pilot is inside the fixed budget.
`cached_verified`: existing source cache and source-training role identities.
Held-source prediction, main training and sealed evaluation: `not_run` by design.

## Findings

| Selected training cohort | Original context-radius decoder | Training-cost-scale decoder |
| --- | ---: | ---: |
|16 nonzero-target rows |98.1793% mean training ADE reduction |98.4397% |
|Same16 plus16 zero-target rows |96.6044% |96.9866% |

These numbers measure fitting the same rows used for training, not a held-set
forecast improvement. All12 models fit these small cohorts well. The original
decoder fits faster early in the trace; the alternative has a slightly lower
final mean error, but is not better in every paired seed. Under mixed targets,
seed17 gets worse with the alternative. No decoder is promoted on this evidence.

The original decoder clips gradients on100% of updates yet fits both cohorts.
The alternative clips on87.18% of nonzero-only and72.03% of mixed updates. Unlike
the previous sampled-gradient report, these fractions count every update.
Persistent clipping therefore is not, by itself, sufficient to explain an
inability to fit. This does not show that clipping has no effect on full-corpus
training or generalization.

The mixed-cohort zero-target harm remains positive:33.888 and21.511
parent-normalized units for the original and alternative decoder. Their zero
baseline denominator still makes relative easy degradation undefined. Large
training improvement is not a certified easy-preservation or deployment pass.

## What This Rules Out, And What It Does Not

- The tested forward/backward path is capable of learning nonzero real-cache
  targets; it is not simply disconnected or entirely unable to fit.
- Including zero targets does not inevitably prevent fitting this selected
 32-row cohort. This does not isolate a prevalence effect: cohort composition
  and full-batch size also change between the16- and32-row experiments.
- The decoder-scale change is not a demonstrated repair of the earlier full
  benchmark. It changes both local sensitivity and a per-row scaling inductive
  bias; the same per-row output range is not an identical conditional function
  class. No held-source score was computed here.
- This is memorization evidence, not proof of causal predictive information in
  the images or of useful motion dynamics. Feasibility selection uses training
  labels and cannot become a held-set filter or an inference feature.

The previous complete60-model negative result remains intact. Its bookstore
training complement had15,430 rows and128,000 sampled draws, averaging8.2955
draws per row. Each row in this microfit receives2,000 full-batch passes. The
two studies differ strongly in exposure, diversity, stochastic-gradient noise
and function-memorization demand. Their percentages cannot be compared as a
matched improvement. Training exposure is now a concrete unresolved hypothesis,
not an established explanation for the full-source failure.

## Next Discriminating Experiment

Keep the whole15,430-row source-training complement, original decoder, objective
and three existing checkpoint seeds fixed. Before another held-site matrix,
compare longer constant-rate continuation with a predeclared rate-decay
continuation, evaluating only complete-training-set loss at fixed milestones.
This can distinguish limited exposure and a step-size error floor from a head
that still cannot use the full training population. Reuse verified parent
checkpoints rather than retrain their first2,000updates; retain both schedules
and all seeds, with no source-held/main evaluation or deployment selection.

If complete-training-set accuracy remains at CV despite that control, investigate
conditional input collisions, target/annotation variability and representation
capacity before another routing search. If fitting improves but held-source
prediction later does not, prioritize generalization and observation support.
Neither branch permits inventing an advantage or changing the primary metric.

## Verification And Status

All12 checkpoint forecasts replay exactly. Six decoder pairs share input/target
identities, full-batch exposure and RNG state. Completed resume preserves37
artifacts and the report, with zero new fits or updates.64 focused tests pass;
the full legacy suite is not rerun. The original generated curve was visually
checked. All processes for this diagnostic have exited successfully.

Report SHA256:
`15ccbdfe8e4c2500072f2e6842f3c83b33de412967e4b39f8624eb28405b76b6`.
[Exact scores](results.md), [curve](training_trace.svg), [verification](analysis.json),
[reproduction](reproducibility.md).

This advances failure diagnosis, not the research success gate. Useful neural
forecasting, independent scene-risk calibration and joint intervention remain
unproved. No deployed model is replaced. The long-term goal remains active and
unmet. No seconds, metric, true3D, foundation, Stage5C execution orSMC claim.
