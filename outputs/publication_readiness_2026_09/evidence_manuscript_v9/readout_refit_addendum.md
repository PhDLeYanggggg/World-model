# Frozen-Readout Optimization Control: 5 November 2026

This is an evidence addendum, not a rebuilt or approved main-paper PDF.
The main draft's prior failed and unavailable results remain unchanged.

We froze three direct-mean neural encoders and refitted their two affine
risk outputs using the exact empirical query weights of their original
training runs. The weighted least-squares systems retain rank 65, with
relative normal residuals below 1.1e-14. Actual float32 fitting MSE decreases
by 1.31%, 4.70% and 2.85%. Encoder parameters, causal inputs, proposals,
normalization, fitting labels and original risk thresholds are unchanged.

On the same seven exposed SOURCE units, known ADE gain over the floor moves
from 4.0912% to 4.3233% (paired change +0.2321 percentage points). The direct
missing-outcome completion-lower contrast is +0.0484 points. This is not a
confidence interval or an independent-test result. Two of three registered
context lower contrasts remain negative, so the prospective expansion
criterion fails.

The original risk/support contract passes 0/17 recording views both before
and after refitting. Known all/easy risk violations remain 13/15. On fitting
data, violations instead rise from 15/15 to 17/18 across 24 repeated recording
views. The refit therefore gives a small descriptive mean-utility gain but
does not fix selected-set risk. It is not promoted to deployment or cited as
a new method contribution. This control narrows an optimization explanation;
it does not show that the encoder is optimal or causal information is exhausted.

Three new closed-form output fits, one exact independent repeat, original
policy inference/readout and exact replay completed. No new encoder updates,
additional seeds or reserved H evaluation were run. The original 24-group
primary remains NA. No identity-fallback certification is introduced.

Evidence: [complete report](../moving_readout_refit_v1/README.md),
[action-flow and risk analysis](../moving_readout_refit_v1/failure_analysis.md),
[frozen protocol](../moving_readout_refit_v1/protocol.md),
[numerical audit](../moving_readout_refit_v1/fit_audit.json),
[method context](../moving_readout_refit_v1/method_note.md).

No metric, seconds, human-gold, true 3D, foundation or safe-world-model claim.
Stage5C and SMC remain off; submission readiness remains false.
