# WEPDTOF: frozen pair-penalty diagnosis

Read the last page of `supplement.pdf` for the movement sensitivity
and the explanation of identical choices. In WEPDTOF the control
removes the entire pair penalty. It is not the cross-term-only
control used in the separate squared-relative formulation.

Run `python verify.py` offline after extraction. It checks the
archive hashes, replays equal-step statistics, verifies the
forecast-only summary arithmetic, and verifies the inherited
evidence. It does not rerun inference or read trajectories.
