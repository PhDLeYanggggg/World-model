# Songdo: score the same observed roster and distinguish the missing labels

I implemented the fixed-budget readout for the four frozen Songdo
proposals. Its input is a set of aligned displacements and one
shared future mask. The scorer itself never opens a dataset. It
requires two distinct edits in the same full query and checks
their displacement radii against the common budget.

## What the readout measures

* Observed ADE, available endpoint FDE, observed ADE p95 and the
  separate complete-roster metrics. A complete pair is not a
  complete roster.
* Direct paired conservative gain against CV, OLS4 and the
  matched controls. The missing-step charge uses the difference
  between the compared paths; it does not subtract separate lower
  bounds. The same query weights and CV error mass are used across
  arms.
* Relative-motion gain on the same selected query roster, with a
  separate available-step score and a missing-pair-step debit.
* The existing all/easy/signed 2% checks, easy/hard observed gain
  and intervention rates. A missing denominator or absent matched
  proposal is not counted as a risk pass.

The source bootstrap averages the three seeds inside each flight,
then averages flights within the linked component before resampling
components. A component with no estimable matched comparison
does not silently disappear. A separate coordinate join checks the
frozen CSV row IDs, actor, drone, visibility and timing before
subtracting each actor's last observed position. The join was
tested with fabricated rows only.

## Verified before opening outcomes

The full engineering rehearsal used the real frozen candidate
packets and fabricated future positions and masks. It processed
200 flights and 600 flight-seed views, including all 573 matched
views across 17 candidate source components. It exercised
111,794 partially masked and 69,959 wholly masked agent histories.
These are fixture counts. The receipt publishes no fabricated
effect sizes and no Songdo accuracy or harm estimate.

No actual future index or coordinates were read. The real
calibration/test assignment and release rule remain to be bound
before scoring; this work does not choose them. No deployment
change, Stage5C, SMC or manuscript edit.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/verify_m3w_songdo_readout_contract.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_future_join.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_matched_proposals.py tests/test_m3w_songdo_first_day.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_pair_support.py
```

Twenty-three related tests passed. [Hash-bound rehearsal receipt](receipt.json).
