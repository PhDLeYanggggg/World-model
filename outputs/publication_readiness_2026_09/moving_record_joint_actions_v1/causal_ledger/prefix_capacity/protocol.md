# Observed Risk And Predicted Prefix Capacity

Posthoc hindsight diagnostic on the previously exposed Moving A roster,
not a learned policy or independent test. Reuse the directional study's
exact frozen direct control: 8 observed / 12 future annotation steps,
native image coordinates, 34 recordings, 14 sources, three frozen
forecaster seeds, 6,952 switches. The forecast bank and original cost
heads are immutable.

For each recording and seed solve four nested mixed-integer problems:
free counts capped at min(2, rows in query minus 1); exact direct counts
per query; exact counts plus two nonpositive predicted totals at the end
of the recording; exact counts plus the same budgets at every prefix.
Use at most one candidate per row, never count an identical baseline
path as a switch. All modes impose the original observed positive all
and easy reference support, 2% all/easy selected positive-harm bounds
including wholly missing-path disagreement, and 2% signed easy net
bound over the full easy reference. Prefix balances start at zero and
are jointly chosen; they are not held to the original path's slack.

Maximize the existing query-balanced paired completion lower relative
to the same direct policy (with CV in the denominator), not an oracle
gain computed by subtracting independent marginal bounds. Compute
paired lower versus CV only as a secondary readout for that witness.
Report infeasible, unresolved and solver-supported witnesses separately.
Eight seconds per mixed-integer solve, zero requested relative gap.
Any numerical incumbent must pass the unmodified original scorer and
predicted budget check. Retain solver objective bounds if available.
Do not report a whole-roster primary on only feasible views. All 102
views remain in the denominator, including zero-switch and no-support
cases. A time limit or numerical rejection is not infeasibility.

Only aggregate witness metrics and statuses are kept; no oracle choices
are exported or used as a training signal. Replay re-aggregates verified
checkpoints; it is not a second solve. No change to official evaluation,
thresholds, deployment, external outcome access, Stage5C or SMC.

Runtime continuation: sampling the original live process showed the
solver in dominated-column presolve far beyond its requested limit.
Keep all completed recording checkpoints and switch only the unresolved
runtime: remove dominated same-row actions (identical count/support,
no better objective and no lower relevant cost), and disable presolve
for the prefix problem. All four problem definitions and the requested
search time limit are unchanged. Keep any timeout unproven.
Sampling the resumed process identified the same dominated-column
presolve delay in the terminal-budget problem. The second continuation
disables that presolve for all modes, also without changing the
optimization problems or recomputing completed recording views.
