# Recording-level replay of the vehicle intervention comparison

Extract these files and run `python -I -B -S verify.py`. No third-party
packages, raw trajectories, images, weights or network access are needed.
The verifier checks the member hashes before reading the scalar inputs.

The inputs are anonymous recording-level quantities from a frozen OpenDD
development comparison, not independent external results. The 8,874
recording views repeat 493 recordings across seven locations, six
calibration choices and three seeds. Identifiers are archive-local
integers. Paths, coordinates, target positions and saved pass flags
are not included.

The replay recomputes the original 2% risk checks from positive harm,
debit, signed easy change and reference masses. It subtracts missing-step
disagreement from paired observed gains and divides pair-motion gain
by the complete pair-step count. For each direction, it recomputes the
tie-complete score prefix on the opposite calibration location and checks
that it matches the frozen threshold. Held scores only apply that
threshold. Empty support does not pass. The 3,000 stored index resamples
draw locations, not repeated recordings. The top-five score diagnostic
is descriptive; it does not choose a different release threshold.

This checks the arithmetic from derived scalar components. It does not
reproduce upstream tracking, trajectory prediction or training.
