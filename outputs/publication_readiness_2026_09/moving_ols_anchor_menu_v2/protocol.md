# OLS-anchored path capacity at exact query counts

This is the same experiment as [the initial registration](../moving_ols_anchor_menu_v1/protocol.md).
A process sample of the first run showed HiGHS repeatedly in mixed-integer
dominated-column presolve well past the configured three-second limit. I
stopped that incomplete engineering run after 13 of 72 view checkpoints;
it produced no final readout. I disabled MIP presolve for both arms in the
new run, keeping the LP, per-recipient dominance removal, all scientific
constraints, seed/recording roles, and configured limits unchanged. No
partial view outcomes are pooled into the replacement readout.
