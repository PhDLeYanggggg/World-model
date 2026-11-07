# What The Timing Experiment Tests

[Mao, Mohri and Zhong (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html)
study regression with several experts, including fitting a deferral rule
after fitting the predictors. The fixed-bank setting here shares that
separation. Our comparison is not their surrogate and does not test
their consistency guarantees. We hold a recording's original number of
interventions fixed and ask whether changing which queries receive them
can reduce baseline-relative ADE under observed recording constraints.

[Yang et al. (SIGMETRICS 2024), Sections 3.2 and 3.3](https://arxiv.org/html/2401.04340v1)
study online allocation with resource replenishment and unknown future
replenishment. Their method prices resources; an extension batches
replenishment to use it in a later frame. Our causal ablation does not
implement or evaluate their algorithm. Its slots come from the current
direct policy, while the other budgets are estimates of baseline-relative
forecast harm. An accounting guarantee for available slots is not a
guarantee on realized forecast harm. Holding unused slots for later is
an ablation, not itself a novelty claim.

The equal-total solve is intentionally stronger than the online ablation:
it sees future labels and can move switches earlier or later. The online
ablation sees neither future labels nor the eventual recording total;
it may leave slots unused. A gap between the two does not identify a
single failure mechanism or justify reporting the hindsight value as a
learned-policy result.
