# Frozen-Action Missing-Outcome Diagnostic

## Question and Registration

The previous source-validation control rejected 41 of 216 candidate views
solely because selected rows had unknown outcomes. This diagnostic asks whether
a deterministic causal disagreement envelope can support any of those same
fixed candidates without treating unknown harm as zero. It does not select a
new policy or read new transfer outcomes. Any subsequent policy comparison
requires a separate registration and action freeze.

Use all 72 already frozen source fits and all three heads (MSE-selected, final,
initial). Reconstruct source-validation IDs, checkpoint partitions, predictions,
actions and the complete old selection readout; require exact agreement with
the committed parent. All learned preprocessing remains that of the frozen
training-source optimization partition. Neither future-label availability nor
this completion screen becomes a per-row inference input.

Source-validation recordings were used for MSE checkpoint choice. They are not
independent calibration. The 12 localities are exposed development sources.
Independent selection, calibration and confirmation remain closed.

## Bound and Estimand

For floor forecast f, neural forecast n and any nonempty subset M of future
labels, the reverse triangle inequality gives:

    |ADE_M(n, y) - ADE_M(f, y)|
      <= mean_{t in M} ||n_t - f_t||
      <= max_t ||n_t - f_t|| <= E.

E is the EXISTING causal envelope: the maximum of floor-neural and CV-neural
step disagreements. Both forecasts use the same origin and native scale one
in this experiment, so translation cancels and units match native ADE.
Using a full-grid MEAN disagreement instead would be invalid for an arbitrary
partial-label mask. We do not substitute full-grid ADE for native partial ADE.

Known partial-label costs and their easy events stay fixed. Only wholly unknown
rows may acquire a hypothetical nonempty label mask and finite target values.
If they never receive labels, the native point metric remains undefined there;
the diagnostic does not fabricate labels. Let S be fixed selected rows, K known
rows, U wholly unknown rows, and D = sum_{S intersect U} E. Then:

    all selected positive-harm ratio <= (H_K,S + D) / R_K,S
    easy selected positive-harm ratio <= (H_easy,K,S + D) / R_easy,K,S
    selected net gain mass >= B_K,S - H_K,S - D

Each denominator must be strictly positive. Unknown references are nonnegative;
for the easy numerator conservatively allow every unknown selected row to be
easy. These are conservative, not necessarily sharp bounds.

For full-population easy ADE degradation, define
N = H_easy,K,S - B_easy,K,S + D. If any unknown row exists, a safe upper bound
is max(N, 0) / R_easy,K. Even an unselected unknown row can add easy reference
mass: when N is negative its ratio can approach zero. With no unknown rows use
the exact signed numerator. Easy rows with zero floor reference can still have
harm; their easy-harm moment must not be discarded.

No strictly positive relative-gain magnitude is guaranteed without an upper
bound on unknown reference mass. Positive lower net mass supports only the sign
for finite completions. No source-to-target transport, population calibration,
physical-safety or full-grid error guarantee follows.

## Fixed Diagnostic Screen

Report a candidate as finite-completion-supported only if both selected-risk
upper bounds are defined and <= 0.02 (numerical comparison tolerance 1e-12),
full easy degradation upper bound <= 0.02, and net gain lower mass > 0.
No threshold search. Original selected-reference denominators and 2% budget
are unchanged. Record old reasons, new reasons, known budget slack and unknown
envelope mass for every candidate; distinguish learned steps from step zero.
Failing this screen means support is insufficient, not proof actual harm is high.

## Verification and Resources

Synthetic/adversarial tests cover partial masks, max versus mean, zero reference,
benefit not cancelling positive harm, unknown unselected easy denominator,
zero-floor easy harm and malformed moments. The real-data run independently
checks aggregate arithmetic and envelope alignment, then replays all 216 rows.
No new checkpoints or row cache; <= 8 MiB new artifacts, CPU4/inter-op1/workers0.
This diagnostic does not invoke or weaken the earlier training disk-reserve
guard. Stop if its own estimated artifact allocation is not available.

Results are fresh_run for reconstruction/bounds, cached_verified for frozen
models/inputs, and not_run for new training, new transfer evaluation and
independent confirmation. Coordinate claims remain image-local detector-silver,
obs8/pred12 at stride12 raw frames, not metric/seconds/true3D/foundation.
Stage5C execution and SMC remain disabled.
