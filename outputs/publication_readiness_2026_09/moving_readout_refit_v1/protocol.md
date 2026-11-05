# Frozen Neural Representation, Full-Fitting Readout Refit

## Question

Does insufficient optimization of the existing neural risk output explain a
material part of the admitted-set error? Do not change the data, causal inputs,
proposal bank, encoder, threshold or original risk/support evaluation rules.
This is a standard least-squares optimization control, not a new loss invention.

The last pilot's mean head produced negative predicted budget excess on its
admitted fitting rows while their observed excess was positive. This is not
solely unseen-source shift. Prior online admission penalties, subset/record
excess losses and conditional event models already failed in related studies;
none is silently repeated here as a new method.

## Fixed Model-Side Intervention

Reuse the three registered mean heads of moving_direct_tail_risk_pilot_v1.
Each has completed2,000AdamW updates. Freeze its entire64-dimensional neural
query-context encoder, input normalization, proposal mask and source roles.
Refit only the130coefficients of its two affine risk outputs by float64
SVD weighted least squares, fixed rcond=1e-10. Save float32 readout weights
and check the loss of the actual unchanged Torch inference path. No threshold
search, additional seed selection, regularization sweep or validation fitting.

For query q visited n_q times in the warm run, and m_q proposed rows, each
proposed row receives n_q / (sum_q n_q * m_q). Queries with no proposals retain
their zero contribution and denominator weight; they are not removed and the
remaining weights are not renormalized. These are the exact empirical sampling
weights of the data-MSE objective evaluated at the final frozen representation,
not a claim that a changing-encoder Adam trajectory minimized that objective.
The two targets stay (H-.02R)/scale and (EH-.02ER)/scale with the old own-B RMS
normalization. Unknown-label contributions retain the old envelope upper bounds
and zero lower reference. They are not counted as observed future labels.

The solve optimizes the data term; it does not reproduce AdamW's implicit path
or decoupled weight decay. Rank, singular values, normal-equation residual,
before/after loss and float32 rounding effects are reported. A smaller fitting
loss is not a safety certificate or proof of better representation learning.

## Source Roles And Readout

Keep the old three contexts, seed17, foreign routerA=1, risk fittingB=0 and
held SOURCE validationV=2. All three refits freeze before Vreadout. V is already
exposed development, not a new independent test. No reserved Moving, DUT,
DroneCrowd or H access. No new full experiment is claimed from this pilot.

Control: original direct-mean head with the original float32 inference.
Treatment: same encoder with refitted readout. Original all-action/proposal-
conditioned controls must reproduce exactly. Decisions precede outcome access.
Compare original known/completion-lower utility, all/easy harm and support,
signed easy-net, switches and same-query/count/amplitude controls. Empty
matched utility remainsNA, not0orpass. Original24group primary remainsNA.

Record separately on B and V: admitted risk error, retained benefit/harm and
rejected benefit/harm for known outcomes, plus unknown selections. Describe
all seven validation source units and17recording views; no new CI or multi-seed
claim. Report whether fitting improvement transfers and whether false-admitted
harm falls, rather than selecting a favorable metric for deployment.

For further compute allocation, require positive completion-lower utility
versus the old mean in each of the three contexts, no increase in known all/easy
violations in each, and actual nonzero intervention in each. This is not a
replacement scientific gate. Deployment remains unchanged regardless of pilot.

## Runtime And Boundaries

Native arm64, CPU4/interop1/workers0; no resource probing. Atomic per-fit
checkpoints, heartbeat and completed-fit resume; one complete independent solve
must reproduce exactly. All inference/readouts replay. Three closed-form
neural-readout fits are fresh_run, not new encoder epochs or NumPy world-model
training. Existing Torch representation and old heads are cached_verified.
The solve is a small local job; no HPC training needed. A queue observation
timeout must not trigger any remote restart or duplicate submission.

No identity-fallback certification, revised2%budget, deleted support rule,
retrospective pass, metric/seconds/human-gold/true3D/foundation claim, Stage5C
or SMC. Native obs8/pred12 in image pixels, not historical SDD rawt50.
