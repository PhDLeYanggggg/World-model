# Failure Analysis

## Structural Zeros Inflate the Easy Ranking Problem

The all-row full harm-event AUROC median is 0.79177; after restricting to
strictly positive causal forecast disagreement it is 0.48578. No future
information defines that restriction. If two forecasts are identical, their
excess error is necessarily zero and the bounded moment head predicts zero.
This explains why the all-row task includes trivial negative cases. It does
not establish a causal mediation fraction or an AUROC significance result.

The motion-only AUROC similarly moves from 0.87301 to 0.55792. High all-row
AUROC is therefore not an adequate gate for learning when a neural forecast
is safe to use. Only the current mean head was tested in these inner folds;
the result cannot rule out information in a better representation or loss.

## Cost Ordering Is Partly Useful

The full model's top10% harm retrieval exceeds envelope ranking in four of six
all-row assignment intervals and three of six positive-disagreement intervals.
The remaining intervals overlap zero; none is significantly negative under
this exploratory bootstrap. Motion-only improves five of six conditional
intervals, despite weak support in 30/72 locality views. These facts must not
be replaced with a blanket statement that the score has no information.

Conversely, the envelope itself retrieves little easy-case harm in its highest
conditional decile (full median zero). Beating this control does not prove
that the lower-score agents accepted by a risk budget are harmless. Top-tail
retrieval is not a replay of the selected action set, and no action set is
newly deployed in this diagnostic.

## Tail Concentration and Magnitude Bias

Full inner-held examples have median harm-event prevalence 2.228%. A median
90.773% of all harm lies in the largest 1% by observed harm, an evaluation-only
oracle statistic. Among positive-disagreement rows the analogous top1% share
is 67.868%. This concentration makes low average fitting loss an incomplete
description of prediction quality.

Full harm coverage is below one in 50/72 views; its range is 0.00207-4.26728.
Motion-only coverage ranges 0.00051-885.02919, with some localities having only
one event. Its median near one is not evidence of calibration. No large errors
are clipped or dropped. Support counts and full per-bin masses remain in the
36 group artifacts, including every weak-support result.

## Fit/Transport and Event Definitions

Original full-model conditional top10 harm capture has median 34.866% on B
and 22.655% on C. These historical role views are dependent. The comparison
does not isolate a single mechanism such as domain shift, capacity or noise.
The new leave-one-locality-out fit repairs one in-sample diagnostic weakness,
not historical exposure of development data.

Inner event cuts, normalization and score bins use three fitting localities.
Original B/C use their original whole-B cut. Do not compare their coverage
medians as a target-matched effect of crossfitting, or pool inner labels to
claim one calibrated event definition.

## What Not to Repeat Without a Distinct Hypothesis

The lineage already includes occurrence/severity hurdle supervision, ranked
hurdle, support-percentile rejection, selected-group penalties, rare-harm
sampling, matched continuation and reference protection. None has established
the full required risk/accuracy result. The present support-restricted event
diagnosis is new evidence; it is not permission to relabel those interventions
as untried remedies. Any next conditional-support repair must retain them as
controls and test a distinct change with the prediction pair held fixed.

## Engineering Corrections During Readout

A reporting-only Path indexing typo was fixed before aggregate generation.
The independent weighted/uniform tail sums differed by 1.37e-12 in one check;
the roundoff tolerance is now 1e-10, with a 50,003-row tied-score regression
test. No trained head, prediction, event cut, threshold or scientific metric
was altered to pass these checks. The replay receipt and scoped suite record
the final verification, not these failed intermediate attempts as completion.

This remains detector-derived image-pixel/annotation-step research. No
metric/seconds, human-gold, physical-safety, true3D or foundation claim.
Stage5C and SMC remain off; deployment is unchanged.
