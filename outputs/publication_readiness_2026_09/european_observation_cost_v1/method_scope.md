# What Observation-Support Factorization Does And Does Not Estimate

The existing completion functional uses nu=1 for any observed requested future
label and causal candidate/floor radiusD. For its harm contribution,

    Z_H = nu*H_observed + (1-nu)*D(X),
    E[Z_H|X] = q(X)*E[H_observed|X,nu=1] + (1-q(X))*D(X).

The same identity applies to EH, while B,R,ER have zero *known contribution*
on wholly unknown rows. This is ordinary conditional expectation. It does not
recover the true unknown trajectory error, full12ADE or partial missing steps.
No missing-at-random assumption is needed for this defined conservative
functional; that is not identification of unobserved physical outcomes.

The implemented auxiliary objective uses the parent's within-query observed
support averaging. Its availability-dependent conditional normalization can
tilt which observed rows receive weight. A finite network minimizing this
weighted multitask objective is not promised to equal the ideal conditional
means above. Observation BCE itself is unweighted by class, with causal legal-
query support; balanced classifier scores would need separate probability
interpretation. None of these fitting statements establishes selected-record
risk calibration, transportability or every-record safety.

The earlier easy-event hurdle models a different event and its negative utility
result remains. This experiment explicitly models missing *released-label*
support and itsDcontribution. Conditional-mixture algebra and adding a logit are
not sufficient method novelty. The primary comparison to the old direct head
is an architecture/objective family effect; only the two new arms match initial
weights and parameters. The isolated secondary identification effect cannot
replace an unsuccessful primary or full original risk check.

## Primary Literature Checked

[Lakkaraju et al., KDD2017](https://www.kdd.org/kdd2017/papers/view/the-selective-labels-problem-evaluating-algorithmic-predictions-in-the-pres)
study evaluation where human decisions determine which outcomes are observed,
using decision-maker heterogeneity. Our detector-release missingness is not
that human-decision mechanism, and no contraction argument is imported. The
official abstract was read; the author-hosted PDF timed out, so full-PDF reading
is not claimed. [Author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915/).

[Boeken et al., UAI2023](https://proceedings.mlr.press/v216/boeken23a.html)
study regression with selection/missing response using privileged information
and conditional-independence assumptions. Their repeated-regression, weighting
and doubly robust identification results do not follow merely from learning
our availability logit. We perform neither imputation nor propensity weighting.
The original paper's missingness setup and positivity discussion were inspected.

[Katsuki and Osogami, ICML2023](https://proceedings.mlr.press/v202/katsuki23a.html)
study untagged, downward-corrupted sensor magnitudes under explicit noise
assumptions. Our binary label-support tags are observable for supervision and
our masked ADE is not assumed to be a downward-corrupted measurement. Their
unbiasedness claim is not transferred to this cost head. The original setup
and assumptions were inspected; no third-party text/code is redistributed.

This is a positioning check, not a new risk theorem, an empirical result or an
independent test. No Stage5C or SMC, no metric/seconds/physical-safety claim.
