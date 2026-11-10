# Supplementary material

Start with `songdo_confirmation.pdf` for the frozen confirmation result.
`original_evidence.zip` retains the reading guide, extended appendix,
and previous offline checks. `songdo_replay.zip` contains the additional
anonymous scalar replay.

Extract this archive and run `python -I -B -S verify.py` from the root.
No packages, trajectories, model weights, or network are needed. The
Songdo replay uses the fixed native cutoffs and reconstructs the score
thresholds from anonymous calibration views. It then recomputes matched
gains, component-level intervals, and actual release counts. It does not
reproduce trajectory inference or the derivation of the native cutoffs.
