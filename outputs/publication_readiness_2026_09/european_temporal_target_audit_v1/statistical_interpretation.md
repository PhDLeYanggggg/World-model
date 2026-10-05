# Statistical Interpretation and Failure Boundaries

This is a controlled, recording-held development probe on 12 exposed localities.
The registered screen passes; policy success and independent confirmation do not
follow. All numerical tables are in [results.md](results.md).

## Unit, Support and Estimand

Bootstrap resamples localities 3,000 times, seed 20261005, after averaging paired
head contrasts within each locality. The 72 forest views and repeated windows
are not independent sampling units. All-row and complete-label contrasts have
12 supported localities; frozen-selected contrasts have 11. There are 596,988
validation row occurrences, 95,455 selected occurrences and 918 unknown selected
outcomes. None of these repeated counts is an independent-test sample size.

The estimand is normalized observed-step error-prediction MSE. It is not executed
trajectory ADE/FDE, a policy improvement percentage, a calibration guarantee,
or selected positive easy-harm risk. Means across differently scaled image-local
records use inherited TRAIN normalization; pooled raw harm masses are descriptive.

## Interpretation Checks

1. **Aggregation and Simpson-type reversal.** All-row signed MSE improves on
   average, but the selected cohort does not beat the leaf-local mean. Locality
   110 worsens against both controls. Report strata rather than infer universal
   gains from the aggregate.
2. **Ecological inference.** A locality-average improvement does not show every
   agent, recording or switch becomes safer. The paired locality is the inference
   unit, not the individual repeated window.
3. **Selection and Berkson-type bias.** Selected rows were chosen by a frozen
   causal policy. Conditioning on selection changes the population; it is useful
   for deployment relevance, but not representative of all trajectories.
4. **Collider conditioning.** Complete future labels can depend on tracking,
   visibility and motion. Complete-label analysis is a sensitivity stratum, not
   a repaired unbiased population or permitted inference filter.
5. **Base-rate interpretation.** Cancellation, occasional positive steps and
   whole-trajectory positive harm are different events. Their prevalence cannot
   be used interchangeably to claim a failure detector or safe policy.
6. **Regression to the mean.** Repeated repair of extreme exposed-development
   failures can improve a later summary without predicting a fresh sample.
   Independent confirmation stays closed pending a frozen supported method.
7. **Survivorship and unknown labels.** Missing outcomes are neither safe zeros
   nor removed from original policy bounds. Step support in a probe is not known
   full-trajectory outcome support.
8. **Multiple comparisons.** Eighteen intervals are reported. Eight prespecified
   conditions form the diagnostic conjunction; intervals remain nominal and are
   not multiplicity- or adaptive-search-adjusted formal guarantees.
9. **Researcher paths and exposed data.** This is one recorded experiment after
   many development investigations. Registration preserves this comparison's
   choices but does not undo prior exposure or establish independent testing.
10. **Association versus intervention.** The matched probe changes target
    resolution under fixed routing and shows predictive structure relative to
    two specified controls. It does not causally prove a neural auxiliary loss
    will change representation, intervention utility or easy harm.
11. **Temporal direction and reverse inference.** Future error labels supervise
    TRAIN fitting and offline scores only. Future masks, endpoints and track
    remaining length are absent from inference. Non-collapse or accurate future
    error prediction would still not prove physical world dynamics.

## Remaining Modeling Limitations

Per-step TRAIN means and per-row observed-step test averaging use different
effective weights when labels are partial. The experiment therefore does not
claim exact optimization of its final score or identification of missing
full-horizon risks. Global TRAIN fallback is explicit: 126 observed validation
step occurrences use it in at least one tree; none lacks global score support.

The reference-error channel has a larger absolute MSE reduction than the signed
channel. Both channels were reported separately to avoid describing reference
scale prediction as successful gain/harm ordering. On frozen selections, the
leaf-temporal minus leaf-constant signed contrast has a CI spanning zero.

Keep the original whole-trajectory harm definition. The cancellation identity is
algebra, not a learned method, safety bound or new modeling contribution. The
standard mean probes alone are not a world-model or publication-level novelty.

The next matched training comparison must retain selected-policy endpoints and
all strong controls. Positive auxiliary accuracy without utility and risk support
is a negative deployment result. No independent-role opening, Stage5C or SMC.
