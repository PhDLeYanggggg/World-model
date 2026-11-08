# Dense Motion Backbone Diagnostic

The previous two-switch ceiling on already opened DroneCrowd calibration
does not rule out broadly applied motion. This follow-up fixes CV and
OLS4 as the only path endpoints. It estimates past-conditioned moments
for the four nonidentity interpolations, with the original CV reference
and 2% rule unchanged. It is not a new formal calibration.

Train three source-balanced fits, 1,000 steps each, on the original
69,746 fitting rows from nine groups. Native arm64 CPU, four threads,
no loader workers. Features use eight positions, backward velocities,
and internal past-only hindcasts. Future labels supervise moments in
the fitting partition only. No fit, price, or threshold is selected on
the 40 DroneCrowd evaluation recordings. Their observation roles remain
the original, already opened calibration roles; 30 confirmation groups
stay sealed.

Freeze all choices before the new readout. Evaluate independent gain
choice, a recording-wide predicted risk price ladder, a fixed half
blend, CV and full OLS4. The matched hindcast control preserves the
joint arm's exact per-query amplitude multiset and ranks recipients by
past hindcast; it need not preserve realized path displacement. Choose
the best predicted feasible positive-gain ladder member, or CV if none
exists. Identity does not pass an undefined risk ratio. Predicted
feasibility is not the actual risk result.

Report paired conservative gain against OLS4 and the matched control,
all-agent gain against CV, actual original recording risk passes,
undefined easy support, coverage, and source-bootstrap intervals after
seed averaging. Original calibration failure is not overwritten.
No deployment, Stage5C or SMC.
