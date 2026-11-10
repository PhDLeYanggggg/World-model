# Supplement: does aligned context improve exchange selection?

Extract and run `python3 -I -B -S verify.py` with no network, trajectories,
model weights or external libraries. This preserves the previous evidence
and adds the map-and-peer ablation. The added replay recomputes aggregates
and location intervals from anonymized grouped scalars, not model training.

All arms use the same 606 OpenDD development views, 780 edits, movement
caps, time credits and original 2% recording rule. The static map describes
vehicle space; peer features describe relative motion, not physical safety.
The misaligned control rotates the same descriptors inside recording and
modality-support strata. The scored location is excluded from fitting,
and every training donor uses a pair-excluded path producer.

At each arm's own release mask, motion-only gives 50 dual passes, map-only
51, peer-only 47, aligned context 47, and misaligned context 56. The frozen
no-exchange reference gives 60. With identical exchange views fixed to the
101 chosen by the aligned model, the respective counts are 46, 48, 49, 47,
and 50. Uniform selection over the legal exchanges on the same 101 views
has 51.21 expected dual passes. The aligned-minus-random equal-location
difference is -0.632 percentage points (95% location interval
[-1.559, -0.026]). Correct alignment does not reliably improve paired
conservative gain over motion or misaligned features at matched swaps.
These are offline development results, not independent confirmation.
