# Coordinating directions at the same recipients and displacement

Extract and run `python3 -I -B -S verify.py` in the extracted directory.
This supplement accompanies the vehicle-direction manuscript revision.
It contains grouped scalar evidence, not trajectories, checkpoints or
raw labels, and it needs no network or third-party libraries.

On 8,874 repeated views from 493 OpenDD vehicle recordings, keeping
the two original actors and every per-step displacement fixed gives
1,431 hindsight views where an angle change improves relative motion
without sacrificing the other checks. In 1,393 of those, one rotation
is sufficient. Requiring two nonzero angles, the causal coupled rule
loses conservative relative-motion gain versus independent angles.
Allowing zero or one change attenuates that loss but does not establish
positive relative-motion gain at the matched rotating actors.

The verifier recomputes the counts and intervals after averaging
calibration directions and seeds within the seven locations. The
archive preserves the previous angular and external-evaluation
evidence. No training or independent confirmation is performed.
