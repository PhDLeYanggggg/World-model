# VisDrone-MOT: use fine boxes only to continue tracks

Pooling two detector scales gave more reference coverage but many more
unmatched histories in the [previous comparison](../visdrone_detector_tiles_v1/README.md).
I tested whether the fine boxes were useful when they could only extend
existing tracks. This is a cross-scale diagnostic, not a reproduction of
[ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php).

The original coarse boxes are matched first. Fine boxes that duplicate a
coarse box (IoU above 0.5) are removed, then the rest can match only an
unmatched, unexpired identity. They cannot start tracks or replace a coarse
match within a frame. One variant requires a coarse match on the previous
sampled frame; the other permits consecutive fine matches. Both replay the
same camera transforms and cost as the coarse control. All 32 detector
checkpoints were checked against the previous freeze; no images or detector
inference were needed this time.

| Rule | Online histories | Complete matches | Complete matches that used fine boxes | Unmatched histories | Fully scored rosters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coarse only | 62 | 21 | 0 | 38 | 0/4 |
| Fine bridge after a coarse step | 91 | 22 | 2 | 65 | 0/4 |
| Consecutive fine bridges allowed | 102 | 24 | 2 | 75 | 0/4 |

The last arm makes 236 fine associations. Only two of its complete matched
tracks actually use fine boxes. Fine continuations also change the positions
used for later coarse association, so the difference in total complete
matches is not the number of directly repaired tracks. No fine-only birth
can explain the larger online roster: an eight-step history in these
eight-frame windows must start at the first frame, when no fine updates
are allowed. Supplemental boxes instead keep more existing identities
observable through the whole prefix. Unmatched does not imply a false
detection; the label scope, geometric tolerance and identity can all matter.

**Decision:** do not adopt this bridge as a camera input improvement or
fit more choices on these four windows. The bottleneck has survived changes
to camera motion, detector scale, and constrained association. A separate
source-level study would need a better-supported tracker and a new,
explicitly held-out protocol. For the paper, the direct question remains
whether joint forecasting decisions improve on independent decisions at the
same budget on supported independent scenes. This training-window diagnostic
does not answer that question or change the existing forecast policy.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_continuation_boxes.py
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_continuation_boxes.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_visdrone_continuation_boxes.py
```

The runner freezes all assignments before reading the training reference
tracks and verifies the per-clip coarse results. No validation or test
labels were opened. The aggregate comparison replays exactly.
[Registration](registration.json), [prefix receipt](freeze.json), and
[per-clip results](readout.json).
