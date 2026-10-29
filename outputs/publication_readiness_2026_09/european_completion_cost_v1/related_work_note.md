# Focused Primary-Source Positioning

This is a focused literature refresh, not a systematic review or novelty
certificate. The current negative experiment must not be presented as a
submission-ready new deferral framework.

Mao, Mohri and Zhong, **Regression with Multi-Expert Deferral**, ICML 2024,
already study jointly trained and fixed-predictor regression deferral, with
surrogate losses and hypothesis-class consistency results for bounded
regression losses. Learned expert routing and a pretrained predictor are
therefore not new here. The paper's stated loss setting does not by itself
certify our variable-envelope, weakly observed trajectory costs or post-
selection per-record risk requirement. This last applicability assessment
is our inference, not an author claim.
[Official proceedings and abstract](https://proceedings.mlr.press/v235/mao24d.html).

Nguyen, Do and Carneiro, **Probabilistic Learning to Defer: Handling Missing
Expert's Annotations and Controlling Workload Distribution**, ICLR 2025,
already address missing expert annotations and workload constraints. It would
be incorrect to claim that considering missing labels in deferral is new.
Our current missing data concern future trajectory ground-truth support,
not merely absent expert decisions on an otherwise labeled classification
sample. These settings must not be conflated. The indexed primary paper
establishes the topic; its detailed method/theorems are not comprehensively
reviewed in this pass because the live OpenReview route returned a browser
verification page.
[Primary accepted-paper PDF](https://openreview.net/pdf?id=zl0HLZOJC9).

The present contribution is only an executed controlled test: retaining
unknown outcomes while adding geometric completion-endpoint supervision.
Its empirical failure and selected-region diagnosis may inform the method,
but do not establish novelty, a new theorem, neural world-model superiority
or independent safety. A future main claim still needs successful held-out
evidence, relevant public comparators and the full approved evaluation.
