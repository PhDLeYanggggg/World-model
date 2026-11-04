# What upward harm refinement can and cannot guarantee

Let each legal action have nonnegative predicted moments B,H,R,ER,EH and causal
rollout-disagreement bound E. With positive correction dH,dEH, use

```
EH' = min(E, EH + max(dEH, 0))
H'  = min(E, max(H + max(dH, 0), EH'))
B'  = min(B, R, max(E - H', 0))
R'  = R; ER' = ER
```

For coherent initial moments in exact arithmetic, H and EH do not decrease,
B does not increase, and the individual predicted admissible action set cannot
expand under the same positive-gain and2%ratio rule. Finite precision and the
original disagreement bound are checked numerically. This is not monotonicity
of joint assignments: a smaller admissible set can cause a different combination
of agents/actions to be chosen. Nor does it bound realized harm or guarantee
calibration, coverage, adequate nonempty support, or utility.

All three fitting arms learn signed two-component residuals, but deployment
applies only their positive predictions. Thus this is not ordinary two-sided
gradient boosting and may retain excess conservatism even where harm was
overestimated. All arms share this restriction; fixed versus refreshed action
weighting is the registered primary comparison. No theorem of multicalibration
or conformal risk control is claimed.

For pooled known rows on an identical selected set, when denominators and
predicted harm are positive, an exact diagnostic identity is

```
(actual H / actual R) / (predicted H / predicted R)
    = (actual H / predicted H) / (actual R / predicted R)
```

The analogous identity holds for EH/ER. Undefined masses remain undefined.
Pooled identities do not establish correctness in every query, recording or
source group. They localize aggregate numerator/reference mismatch, not a
unique causal mechanism or independent population guarantee.

The full-information conditional mean, if correctly learned with all legal
action labels, can support expected decision costs. These failures do not show
that squared-error regression is inherently decision-inconsistent. The study
instead asks whether a small, action-conditional function class and refreshed
fit weights correct the selected-region magnitudes that the earlier networks
missed. Negative outcomes must remain negative.
