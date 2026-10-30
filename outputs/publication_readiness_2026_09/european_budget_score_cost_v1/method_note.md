# Why Supervise The Deployed Budget Scores?

## A Counterexample, Not A Performance Claim

Suppose the true benefit,harm,reference,easy-reference and easy-harm moments
are(0.2,0.1,1,0.5,0.05). A predictor gets both harm moments exactly right
but predicts references10and5. The fixed decoder sees positive net benefit
and harm/reference ratios1%,so it switches. Both true ratios are10%,not2%.
This construction respects nonnegative,nested easy moments. It demonstrates
that penalizing harm underprediction alone cannot address all optimistic
decisions. The regression test exercises the actual unchanged decoder and
the score diagnostic on this example.

## Retaining Identification While Coupling The Error

For the all-case score,the prediction error is

```
e_q = (H_hat - H) - 0.02 * (R_hat - R)
```

Thus a negative e_q can arise from low predicted harm,high predicted
reference error,or both. The analogous easy score is included separately.
The upper_score auxiliary loss upweights this negative residual by4while
the symmetric_score control does not. The original five-moment regression
and ranking terms remain,so this is not a score-only model with unidentified
components. No additional neural parameter or prediction-time threshold is
introduced.

The new loss supplies gradients to both the numerator and denominator.
It does not prove either component is a confidence bound,that selected
recordings obey the budget,or that the modified policy improves utility.
Those are empirical questions in the complete registered comparison.
The toy counterexample is engineering/mathematical evidence,not real-data
performance or a guarantee. It also does not prove denominator inflation
is the only explanation of the previous study's observed failures.
