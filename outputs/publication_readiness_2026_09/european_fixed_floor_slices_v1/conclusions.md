# What the Source-Gap Decomposition Establishes

## Material Passport

Fresh diagnostic readout of frozen, hash-verified models and actions. No new
training or deployment. The protocol was committed as `111157b4` before the
slice readout. All108source groups, three forecaster seeds and12development
localities are retained. Independent selection/calibration/confirmation remain
closed. This is exploratory diagnosis, not a new safety primary.

## Findings

1. **The current radial support guard is insufficient.** All selected harm is
   inside that guard because outside rows already fall back. Held score error
   remains0.1836 inside; the source-pooled selected harm ratio is2.1438%
   [1.5145%,2.8551%]. Widening/removing an outside-domain guard is not the repair.
   Being inside a marginal feature radius does not establish conditional overlap.

2. **Risk varies strongly with observed motion and predictor disagreement.**
   The low/medium/high disagreement slices have selected harm0.9562%,3.2378%,
   and5.2664%. The high-bin interval is[2.6756%,8.0658%]. Low mean-turn samples
   have7.5134% [4.6536%,10.5933%], versus0.9857% in the high-turn bin. These
   are fitting-defined bins, not thresholds optimized on held labels. Their
   association is not proof that turning causes safer predictions. Detector
   jitter, motion magnitude, neighborhood extent and source composition can
   jointly affect these proxies.

3. **Missing labels are not a complete explanation.** Complete-label samples
   contribute61.2841% [52.5767%,68.1811%] of observed selected harm. Their
   selected harm ratio is2.2698% [1.6396%,2.9190%]. Partial labels contribute
   the remaining38.7159%, with2.0060% [1.3599%,2.8534%] selected harm. The new
   objective's squared-error difference is+0.0198 on partial labels and-0.0021
   on complete labels, but these post-hoc intervals are not multiplicity-adjusted.
   Future label completeness is never an inference feature or deployable filter.

4. **A tiny denominator is not the only failure.** Low reference-error rows
   contribute23.8281% of observed harm, middle rows49.6338%, and high rows
   26.5381%. The middle error bin still has2.6307% harm. Ratios must be read
   together with normalized harm mass and selected counts; unknown labels and
   zero denominators cannot be recoded as safe.

5. **Useful actions still exist within the allowed causal pool.** Eligible
   oracle benefit is16.2717% [13.2406%,19.0725%] of floor error, compared with
  21.2532% unrestricted diagnostic oracle benefit. The policy captures only
  1.0804% [0.6563%,1.5772%] of available positive-benefit mass. This identifies
   opportunity, not learnability or a deployable result.

6. **Not every axis was informative.** Tied fitting quartiles put every row
   into the nominal high clipping-fraction bin. This does not mean every row
   is clipped. The neighbor-high stratum also has incomplete fixed-locality
   support. Empty bins remain explicit; no favorable subset is silently retained.

## What Changes Next

The next narrow hypothesis is that the risk head needs explicit nonlinear
motion/disagreement descriptors rather than only an implicit encoding of these
relations in flattened histories. A matched feature-only comparison should
append continuous causal descriptors, retain the SAME prediction/floor/utility,
loss, initialization of shared weights, draws, source roles, budget and steps,
and compare against the frozen380-feature control. It must test risk and
same-query-count ranking, not merely lower score MSE or lower coverage.

This is a hypothesis selected from development evidence, not an independent
claim. Do not deploy a low-motion threshold discovered in these slices. Do not
add future completeness or realized reference-error bins to inputs. Do not
increase risk tolerance to make a gate pass. A label-quality sensitivity can
be secondary, but dropping partial labels cannot be presumed to solve the
complete-label failure. No new feature model has been trained in this record.

The registered parent result remains negative:14empty dependent views,
79/216over-budget views, and no supported equal-count ranking improvement.
Pooling within locality here does not repair that primary. Deployment remains
unchanged. Full legacy tests and a cold raw-data rebuild were not run.

Image-local detector silver, obs8/pred12 rawstride12 only. No metric, seconds,
human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.
