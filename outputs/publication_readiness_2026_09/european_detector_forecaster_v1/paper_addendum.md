# Producer-Side Observation Context: Development Evidence Addendum

This is a scoped experiment addendum, not a submission-ready manuscript or
an independent-test result. Numerical status is governed by the fixed readout
and separate replay receipts. The core paper still concerns useful and safe
neural intervention over strong motion references, not a new projection layer
as a standalone world-model contribution.

## Research Question

Earlier detector-history inputs were supplied to intervention-cost heads, while
their underlying trajectory producers remained unchanged. The present control
asks whether supplying those same causal measurements to the producer improves
its actual trajectory outputs. Proposal accuracy and safe use of a proposal
are evaluated as distinct questions.

## Method

Let z_t denote the original ego trajectory token and b_t the eight detector
features at observed time t. The controlled arm embeds the ego token as
E(z_t) + m_ego + W b_t, with W initialized to zero. The eight ego tokens,
72 aligned past-neighbor tokens, Transformer backbone, causal physical
reference, query times, observed-motion output bound and masked training loss
remain unchanged. W has 512 parameters; each producer has 89,026 parameters.

Past detector features combine log box width/height, raw confidence,
regularized aspect ratio, relative height change, body-relative center steps
and consecutive-box overlap. Their fitting-locality normalization is independent
of future-label availability. Future targets are loss/evaluation labels only.
No future endpoint, central velocity, test-endpoint goal or held normalization
statistic enters the model. A detector-silver export is not certified live
sensor-as-of data.

Four equal-size branches are retained: disabled, latest-box-only, causal-key
noise and full detector history. Equal size does not imply equal information
rank. All 18 original producer folds/seeds are trained for every arm, using
4,000 updates each, source-balanced batches of 64 and the original optimizer.
The complete original 318,969-row,163-record producer population is retained.
The readout is limited to 149,510 old controller-fitting queries, each outside
the evaluated producer's own fitting localities. The other 89 recordings are
not newly scored as controller-held outcomes.

## Results And Uncertainty

Known masked-ADE gain over the producer's fitting-selected causal reference
is 4.923% (disabled),5.961% (current),4.841% (noise),8.811% (history).
The paired history increment over disabled is +3.888pp, 95% locality-bootstrap
CI [+2.246,+5.655]. History also exceeds both current-only and noise controls
on these known labels. Three actual producer seeds are retained; 3,000
bootstrap draws resample 12 locality clusters after fold/seed averaging.

However, the primary completion-lower contrast is -11.355pp, CI
[-27.809,-0.344], with unknown outcomes retained. This is a conservative
bound, not an identified observed degradation. Complete-population gain is
not established. The complete locality roster has an undefined FDE summary
because locality082 has no final-step labels in the allowed readout. Secondary
complement evidence is weaker: history-minus-disabled +1.231pp,
CI [-3.005,+4.786]. No secondary replaces the registered primary.

A supplementary post-readout paired comparison with each fixed physical
method also limits the claim. History-minus-damping0.97 is +2.375pp,
CI [-2.731,+6.710], on the same original reference-mass denominator.
This does not prove superiority over every strong fixed baseline. The
comparison is descriptive, not used for model selection or the primary gate.

Positive average accuracy does not establish the complete-policy risk rule:
easy-subset mean gain is only +0.552%, CI [-4.154,+6.631], and individual
localities degrade substantially. This raw-proposal experiment learns no new
intervention policy. It authorizes neither deployment nor opening independent
confirmation roles.

## Relation To Prior Work

Noisy observation histories and temporal attention are established topics.
For example, [Monti et al., identified arXiv v1](https://arxiv.org/pdf/2203.04781)
study observation count and noisy tracking inputs. Our recorded reading scope
does not claim a matched implementation of their method or an exhaustive
comparison to noise-aware prediction. The [reading note](literature_position.md)
separates full sections read from abstract-only material. The present branch
is an empirical diagnostic, not sufficient architectural novelty by itself.

## Limitations And Required Paper Evidence

SOURCE is detector-silver midpoint-pixel data with obs8/pred12/stride12 raw
frames. Partial labels are not a full future path. No verified homography,
effective seconds, metric geometry, human-gold target or true-3D state is used.
Past box cues are bundled, and no independent scene/goal/JEPA contribution is
established. The original stationary bound remains, but this readout has no
exactly stationary ego histories, so it is not an observed explanation here.

Historical development exposure and extensive prior model development limit
confirmatory interpretation of the locality intervals. The next paper-level
evidence needs learned gain/harm control on the frozen bank, scene-level joint
decision controls, independent risk calibration/confirmation and appropriately
matched published predictors. Public source/configuration and exact local
replay are not yet a complete anonymous from-scratch asset package.

See [full fixed readout](readout.md), [failure taxonomy](failure_analysis.md),
[model/data card](model_data_card.md) and [reproduction guide](reproduction_zh.md).
No Stage5C execution, SMC, deployment or submission-readiness promotion follows.
