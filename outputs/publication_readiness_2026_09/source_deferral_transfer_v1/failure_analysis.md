# Failure Analysis: Candidate Utility and Cost Transfer

## Observed Versus Hypothesized

| Mechanism | Evidence | Conclusion |
| --- | --- | --- |
| Candidate dynamics remain weak | Both new raw proposals and dense control lose to CV; cost-supervised proposal-minus-dense CI crosses zero | Supported for these fixed endpoints; not proof that neural prediction is impossible |
| Pure expected-cost routing collapses | All three seeds produce baseline-only actions on training and held source | A safe no-change action, not learned predictive improvement |
| Cost-supervised routing does not transfer | Mean training gain +0.233917%, held -1.245558%; all three held seeds negative | Supported failure of this complete trained package |
| Training costs are optimistic for gate learning | In-sample cost supervision; intervention 61.002% train versus 62.985% held while gain reverses | Plausible; needs out-of-fold candidate costs, not identified as the unique cause |
| Moving-target gate lag | Earlier training study: proposals improve late while plain expected-cost scores stay negative | Plausible optimization mechanism; requires frozen-candidate control |
| Easy cases are harmed | Cost-supervised hard action adds 0.01581748 native-pixel ADE at zero CV error | Original percentage safety criterion cannot be evaluated; no safety pass |
| Window weighting alone explains failure | Equal-recording -3.354742%, equal-scoped-agent -1.460942%, worst recording -13.996053% | These alternative weightings do not reverse the negative result |
| Exact input aliases make the task broadly impossible | 153 duplicate-input training rows, 38 conflicting rows; no certified zero-optimal conflicting group | Not supported as the dominant cause; approximate ambiguity is untested |
| A few outlier gain labels explain collapse | Prior training diagnostic found only 3-4 out-of-unit targets and small Huber/mean difference | Not supported as the principal explanation; plain objective does not use that Huber target |
| Runtime or failed replay explains the score | All nine forecasts replay exactly; 313 artifacts unchanged; finite and bounded outputs | No such defect found in these checks; tests do not establish scientific efficacy |

## What This Does Not Establish

The hard-action improvement over dense is a useful error decomposition, not a
success claim. A lower error than an inferior neural comparator is insufficient
when the intended fallback already predicts better. A sigmoid score was not
independently calibrated. Constant rejection therefore cannot be interpreted as
successful detection of individually unsafe cases.

Bookstore was excluded from current fitting but explored in earlier work.
Seven recordings and three seeds do not create seven or three new independent
physical sites. Recording-block intervals quantify a conditional diagnostic;
they do not erase prior test exposure. The main equal-site 8-to-12 evaluation
and its sealed roles remain unchanged.

The cost-supervised training arm simultaneously adds signed-gain supervision
and a proposal-loss term. Its contrast is a package effect, not isolated proof
that either component caused the observed behavior. No threshold was tuned
after this fixed readout, and no easy row or losing scene was removed.

## Literature Boundary

Joint and two-stage regression deferral already exist. Mao, Mohri and Zhong
develop consistency-supported surrogates for bounded regression costs, including
instance- and label-dependent costs. Adding a gain head or a frozen predictor
is not, by itself, a new M3W contribution.
[Official ICML 2024 paper page](https://proceedings.mlr.press/v235/mao24d.html).

Li et al. study learning the predictor on all data before rejection, with
consistency under weak realizability and prediction/calibration error terms
beyond it. Their assumptions do not automatically cover our ADE objective,
baseline-dependent costs or correlated windows. This is motivation for a
controlled optimization comparison, not a theorem explaining our experiment.
[Official AISTATS 2024 paper page](https://proceedings.mlr.press/v238/li24g.html).

## Next Falsifiable Check

Use only the four outer-training source sites to construct inner held-site
candidate forecasts. All target-dependent preprocessing and parents must obey
the same exclusion. Test whether useful candidate gain survives honest
out-of-site prediction before learning a new cost head. Preserve all folds and
seeds; do not choose the best bookstore threshold or family. If candidates are
still unhelpful, stop routing optimization and investigate predictive inputs
and trajectory supervision using training-side controls.

That check has not run. Current evidence justifies neither deployment nor a
new positive world-model or publication-readiness claim. Stage5C/SMC remain
off. Dataset annotation pixels, past-normalized coordinates and raw frames only.
