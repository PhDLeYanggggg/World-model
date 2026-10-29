# Protected Harm Branch: Exact Experimental Method

## What Is Learned

The frozen forecast producer and strong motion floor define supervised cost
targets from future labels. Those future values are never supplied at inference.
The causal feature vector has380 dimensions; its source-specific normalization
uses the original inner-fit recordings only. The cost head estimates five
nonnegative quantities: benefit B, harm H, floor error R, easy floor error Re,
and easy harm He. Easy labels are supervision, not an inference-time filter.

Let E be the causal error envelope. The existing head enforces B+H<=E, Re<=R,
and He<=H. A frozen shared-utility anchor supplies B0,H0,R0,Re0,He0. A separate
encoder, initialized as a copy of the anchor encoder, produces two odds shifts.
Each shift is bounded as d=5*tanh(a/5). With C=E-B0 and q=He0/H0, the equivalent
ideal-arithmetic update is:

    H  = C * sigmoid(logit(H0/C) + d1)
    He = H * sigmoid(logit(q) + d2)

The implementation uses incremental odds identities to preserve the zero-head
initialization and avoid direct logit infinities. Zero/one boundaries use
float-safe denominators and clamps. The bounded odds range can limit correction
of an extremely confident wrong anchor; a successful optimization run alone
does not rule out that restriction. Exact B/R/Re protection is checked on real
fit/held rows; net benefit B-H can and must change as H changes.

## Objective and Matched Controls

Keep the inherited primary moment/signed-score regression and decision regret.
Risk arms also optimize source-specific empirical all/easy excess using dual
multipliers. Utility continuation fixes those multipliers at zero. The2% budget,
temperature schedule, positive-harm definition, RMS and constraint scales are
unchanged. Every arm receives the same source/query draws and fresh AdamW state.

Isolated risk freezes the full anchor. Joint risk and utility continuation
train the same total architecture including the anchor primary head. Unused
auxiliary weights remain frozen. This controls total architecture and exposure,
but not number of trainable parameters. No gradient-conflict identification,
formal safety guarantee or novel-method claim follows merely from this design.

## Inference and Honest Support

Eligible rows are determined solely from observed motion and causal support.
The unchanged hard rule switches only when predicted B-H>0,
H-0.02R<=0 and He-0.02Re<=0. Otherwise it retains the motion floor. Neither known
outcome availability nor oracle gain filters the inference rows. A source not
represented by the registered preprocessing is rejected rather than silently
mapped to a familiar domain.

Unknown outcomes are retained with finite completion envelopes. Same-count
comparisons match within recording and frame using causal scores and stable row
indices, not future ranks. Full-policy risk remains the safety criterion; a
same-count diagnostic cannot override an unsafe full policy. Every context,
source and seed is retained, including empty selections and undefined ratios.

All evaluated held recordings are previously exposed TRAIN-internal material.
This is a falsifiable local mechanism test, not unseen-domain generalization.
No trajectory model, JEPA pretraining, latent generation, Stage5C or SMC is run.
