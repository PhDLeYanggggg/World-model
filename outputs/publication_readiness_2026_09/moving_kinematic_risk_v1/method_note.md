# Model Change And Claim Boundary

Let x be the original standardized causal input and phi(x) the new64-field
motion basis, standardized using only critic-fitting sources. The common
hidden layer changes from

    h = SiLU(Wx + b)

to

    h = SiLU(Wx + b + U phi(x)).

The action one-hot is included in x. U starts at exactly zero. The existing
query-mean interaction block and two signed risk outputs are unchanged. W, b,
the output layer and U are optimized in the treatment; the matched control
has the same parameters and optimizer but receives phi=0, so U remains zero.
Both arms reset AdamW identically and continue from the same trained weights.

The original outcomes used only for training/evaluation are positive harm H,
reference error R, and their easy-event counterparts EH and ER. The two
supervised outputs remain H-0.02R and EH-0.02ER, with the original training-only
normalizers. Source/query weighting, proposal masks and unknown-label bounds
are unchanged. The decision still requires nonpositive predicted excess and
positive predicted support. No new threshold or tolerance is optimized.

This is a deterministic transformation of existing past data and legal model
rollouts. It does not supply a target endpoint, central velocity, test-derived
normalizer or new scene label. A rollout endpoint is a causal forecast, not
an observed future. Closest-approach features extrapolate the last backward
relative velocity over a bounded native-step horizon; they are neither measured
collision times nor a physical-validity guarantee.

The comparison answers a narrow causal-design question: holding training and
evaluation fixed, does making these nonlinear motion relationships explicit
improve the learned admission decision? Equal nominal parameter count does
not imply equal effective capacity because the masked branch cannot learn.
It does not isolate temporal reasoning, neighbor reasoning or any individual
feature. Those would need separately frozen ablations after an aggregate gain.

Neither the feature basis, mean-pooling block nor ordinary MSE is claimed as
a novel method. A successful local comparison would justify broader controlled
training, not by itself establish an A-conference contribution. Publication
claims still require a useful original decision method, matched intervention
controls, original risk compliance, statistical evidence and independent
source evaluation. Failure narrows this representation hypothesis, without
proving the absence of learnable causal information.

The current exposed-source population is not reopened independent testing.
Its unavailable primary/support cases and previous failures remain unchanged.
No identity certification, deployment upgrade, Stage5C execution or SMC.
