# What The Paired Contrast Does And Does Not Test

This analytical note documents the registered losses; it changes neither
training nor evaluation. It is not an additional arm, tuning criterion or
empirical result. The accompanying synthetic derivative tests do not establish
real-data safety or downstream value.

For one observed harmful/beneficial pair, let p and n be the scalar maximum
scores and w>0 its fixed cost weight. The registered extra losses are

```
point(p,n) = 2w[softplus(-p) + softplus(n)]
rank(p,n)  = 4w softplus((n-p)/2)
```

At (p,n)=(0,0), both have value 4w log(2) and gradient (-w,+w). Their
score-space Hessians are

```
H_point = (w/2) [[1, 0], [0, 1]]
H_rank  = (w/4) [[1,-1], [-1,1]]
```

Both therefore have the same local curvature along the difference direction
(1,-1). The rank term has zero curvature along the common direction (1,1),
whereas the point term anchors both scores. This is a score-space calculation,
not the Hessian of the full neural model: max-composition can be nondifferentiable
and the retained base loss supplies other gradients. Warm-start scores are not
assumed to be zero. Equal null value/gradient does not mean equal realized
gradient magnitude or objective throughout training.

For every common shift c, rank(p+c,n+c)=rank(p,n). For example, the score pairs
(2,-2),(-8,-12),(12,8) have identical ranking loss and ordering, but the fixed
zero boundary respectively accepts only the useful proposal, both proposals,
or neither. Thus perfect ordering alone cannot identify safe acceptance.
The unchanged supervised risk and joint-decision anchors remain necessary;
they are not themselves a statistical guarantee. Positive fitting AUROC, low
ranking loss and exact recovery cannot replace the original 2% full-policy
risk, easy, reference and support requirements.

The experiment isolates relative discrimination from a matched pointwise
contrast under fixed pair construction and budget. It does not introduce a
new ranking loss, calibrated probability, theorem of safe deployment or neural
trajectory forecast. Unknown outcomes and zero targets do not form beneficial
pairs. No held outcome determines the loss coefficient or inference threshold.
