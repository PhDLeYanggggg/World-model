# Why Causal Retrieval Did Not Repair Safe Decisions

## The Controlled Change

I replaced the parametric risk readout with a fixed local risk estimator, not
the trajectory predictor or evaluation rules. The query's whole source was
excluded. Each query used 32 source/agent buckets and at most one window from
each bucket. Original120 features, two signed risk targets, source/query
weights, fitting-only normalization and the zero decision boundary stayed fixed.

The study covers all 48,061 known own-B proposals in eight source-held folds.
The 1,274 unknown proposals remain outside supervised scoring, explicitly
audited. This is exposed development data, not independent confirmation.

## Observed Failures

1. **Local mean risk is not reliable enough for admission.** Decision-cost
   reductions versus shared64 are -4.481672%, -121.691096% and -2.568241%.
   Harmful admissions change from 0/1/0 to 107/723/521. More proposals are
   admitted, but their harm outweighs the recovered useful decisions.
2. **Similarity does not make risks transferable.** Source076's [0.25,0.5)
   nearest-distance bin contains 845 rows: weighted predicted signed-risk
   exceedance is 6.535%, versus 31.409% observed. In [0.5,1), 1,893 rows have
   8.243% predicted versus 48.334% observed. Source067 overestimates the same
   event instead. These are fixed-bin diagnostics, not new distance thresholds.
3. **Better regression is an insufficient objective.** Context2 risk MSE
   falls 33.420%, but its decision cost worsens 2.568%. A constant fitting
   prior already reduces MSE by 30.094%, while admitting nobody. Neither
   result establishes usable safety or independent predictive improvement.
4. **Source coverage is thin.** Context0 has only one donor source per fold;
   the other contexts have two. There are 112-1,036 conservative identity
   buckets, not that many independent environments. Source076 has 112 donor
   buckets and a median effective weight count of 15.446 among its 32 selected
   references. Query/source weights can concentrate individual matches.
5. **Support is still a separate problem.** Source053 admits 27 known proposals
   without selected easy-reference mass; source065 admits none. Their easy
   ratios remain NA. Other source-specific descriptive easy-harm ratios are
   high. None is a substitute for the original full-policy support and risk
   evaluation. No identity-based exemption is applied.

## What Worked, And What It Does Not Prove

Correct matching lowers decision cost relative to the fixed misalignment by
11.453%, 20.948% and 15.575%. It also lowers harmful-admission counts at the
same unweighted number of admissions in each source. Thus this particular
matching uses some useful history information; the result is not simply
equivalent to an arbitrary assignment of the predictions.

This is one null permutation, not a statistical significance result. Query
weights need not remain identical under permutation. The comparison does not
prove causal sufficiency, calibrated uncertainty, an irreducible error floor,
or safe generalization. Mixed labels among 32 neighbors are expected even
without deep ambiguity. The fixed information screen fails all three contexts.

## Next Model Question

Do not expand this k32 estimator into a deployable policy, search held-source
distance thresholds, or claim another wider forest will fix the problem.
The next targeted check is whether the harmful near-matches differ in causal
forecast amplitude, ego motion, peer-relative motion or scale context that the
current Euclidean representation obscures. Attribute those blocks on existing
own-B matches first; compare prior negative relational/ego-frame experiments
before registering one distinct learned relational-risk change. Any learned
metric must be fitted on donor sources only. This is a next action, not a
completed repair or a license to reopen held confirmation.

The shortfall may involve representation, source-conditional label shift,
thin source support and partial supervision. This experiment cannot identify
a single cause or prove that world-model dynamics are impossible. No new
neural trajectory model was trained here; deployment and the original24-group
primary's NA status remain unchanged. Stage5C/SMC remain disabled.
