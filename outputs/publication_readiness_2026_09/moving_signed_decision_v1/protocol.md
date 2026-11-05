# Signed-Cost Decision Learning: Prospective Comparison

## Scope And Question

I test whether learning the sign of expected budget excess with a cost-weighted
surrogate improves admission compared with ordinary squared-error continuation.
The original risk, easy-net, support and empty-control rules remain unchanged.
No identity certification. No new test, no changed scientific endpoint.
This is neural risk-head training, not a trajectory/JEPA training experiment.

The prior source-held error experiment is negative. Its selected error MSE
falls but full-fitting mistakes increase and held utility deteriorates. The
fresh B-only boundary audit shows that mistakes are not merely roundoff at
zero. In the third context 590 of992 false-safe easy output channels have
absolute normalized scores between0.1and1, and three exceed1. Counts and
cost-weighted error also differ: held-source discovery has higher easy sign
regret in two contexts but lower regret in the third. These are repeated
fitting channels, not independent people or held efficacy evidence.

## One Changed Ingredient

Both arms use the same 32,130-parameter kinematic neural head, frozen warm
weights, own-B preprocessing, source-balanced query sampler, AdamW reset,
learning rate0.001, gradient clip5, width64 and2,000updates. Inputs, proposals,
unknown-outcome bounds, support costs and zero admission thresholds do not change.

- Control: continued MSE on signed all/easy budget excess.
- Treatment: for observed outcomes use `4*abs(y)*softplus(-sign(y)*s)`;
  for unknown outcomes retain the identical bound-aware MSE.

Here `y=(harm-0.02*reference)/own_fitting_normalizer`, separately for all/easy.
The existing causal past scale and own-fitting RMS remain fixed in both arms.
There is no inferred error-priority mask, discovery teacher or class rebalancing.
Each actual row/action cost supplies its weight; zero costs have zero surrogate.
Temperature is fixed1, not tuned. Multiplier4 gives the same derivative at
score0 as MSE: `-2*y`. It does not ensure equal gradients away from zero.

For a fully observed conditional cost distribution, write
`A=E[max(y,0)|x]`, `B=E[max(-y,0)|x]`. Conditional loss is
`4*(A*softplus(-s)+B*softplus(s))`, minimized at `log(A/B)` when A,B>0.
Its sign is the sign of `A-B=E[y|x]`, rather than the majority sign-event
probability. One-sided/zero-cost cases have infinite/indeterminate optima.
This elementary population derivation is NOT finite-data calibration, a safety
guarantee, or a proof for the mixture with unknown-bound MSE. It is also not
a new-method claim. Scores are not predicted expected-harm magnitudes.

Earlier cap-event BCE estimates event probability; hurdle losses factor
occurrence/severity. Earlier online-admission objectives use soft decisions
and dual penalties. The previous priority trials multiply MSE on selected
mistakes. This test instead applies a magnitude-weighted sign surrogate to
all observed legal proposals, without an auxiliary output or changed threshold.
It tests a distinct objective, not a novel architecture or guaranteed repair.

## Roles And Budget

The same three registered seed17 contexts are retained. A(part1) supplies the
frozen foreign proposer; B(part0) trains the head; exposed SOURCE V(part2)
is evaluated only after all six final models freeze. B has2/3/3 sources.
Whole-source leakage and fitting-only normalization checks are inherited.
The old original producer/easy-cut lineage is verified, not retrained here.
No future endpoint, future validity mask, test goal or central velocity enters
inference. Unknown targets are never recoded as observed safe examples.

Six real fits give12,000primary updates. A full treatment refit adds2,000.
Both arms also receive a100-update resume check:50+50 must equal100direct,
400updates including repeats. All three fresh control models must exactly
reproduce previous MSE model weights/sampling. Checkpoints every100updates,
heartbeat every100updates, CPU4/interop1/workers0, nativearm64,12h guard,
5GiBdiskreserve,512MiBprivateartifactcap. No NumPy substitute for training.
No data cap reduction for slowness. No new reserved-domain or H readout.

## Analysis And Decision

Keep all seven exposed SOURCE units and17recording views, one producer seed.
Recompute old cached-control metrics and new inference/actions/metrics; replay
the latter exactly. Report equal-source known ADE/hard gain, switch rate,
positive-harm/easy-risk violations, undefined supports and worst easy-net.
Paired missing-label completion intervals are NOT sampling confidence intervals.
Empty same-count utility controls remain NA. Original24group primary remainsNA.

The same prospective expansion screen requires positive paired completion
lower gain versus MSE in EVERY context, nonzero treatment actions, no increase
in known all/easy violations or undefined selected/full-easy views, and a
positive aggregate completion lower contrast versus affine. This is only an
expansion screen, not the original safety gate or permission to open H.
No context, epoch, threshold or arm is selected after inspecting V. Report the
complete result, even if all-fallback or negative. Deployment stays unchanged.

Posthoc fitting diagnostics may compare MSE, cost-weighted sign regret,
false-safe/false-reject counts and mass, and observed versus bound losses.
They cannot redefine the endpoint or justify a favorable-source subset.
Reserved20Moving groups, DUT, DroneCrowd and independentH stay closed.
Native obs8/pred12 imagepixels: not verified seconds/metric/true3D/foundation.
No human-gold label claim. Stage5C/SMCoff, submission_readyfalse.
