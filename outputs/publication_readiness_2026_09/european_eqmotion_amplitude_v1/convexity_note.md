# Why Fixed Support Helps, And What It Does Not Prove

For a nonempty fixed future mask, available-label ADE is convex in predicted
positions. Write b for the causal floor, p for the fixed proposal, and
p_alpha=(1-alpha)b+alpha*p. Convexity gives

    C(p_alpha)-C(b) <= alpha*(C(p)-C(b)),  0 <= alpha <= 1.

Taking positive parts gives H_alpha <= alpha*H_1. For constant positive alpha
the mathematical nonalias intervention support is unchanged, so the selected
known all/easy reference denominators do not shrink. Unknown-row path
disagreement is exactly alpha times its original value. Consequently both
completion-risk upper ratios are at most alpha times their alpha=1 values
where their fixed denominators are positive. Signed easy-net degradation has
the corresponding upper inequality; its numerator may be negative.

This is a direct convexity argument, not a new theorem or distribution-free
generalization result. Numerical aliases must still be counted. It does not
hold with an unchanged denominator after arbitrary row pruning. It does not
fill partial labels, predict actual unknown futures, or supply a causal
test-time value for a future-defined reference mass.

If every required reference denominator is positive, arbitrarily small
amplitude can in principle reduce these finite-sample upper ratios. That does
not guarantee a useful gain, a positive lower utility bound, or transport to a
new recording. SOURCE-derived maximum safe amplitudes would use future labels;
they are not legal inference features or independent deployment thresholds.
The registered ten-point curve therefore remains a diagnostic, not a calibrated
deployment policy. No risk tolerance or reference definition has changed.

The supplementary randomized tests check this algebra under complete and
partial fixed masks. They are engineering checks, not real-data efficacy.
