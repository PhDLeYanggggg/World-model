# Evidence Manuscript v95

I added the [fixed-query recipient experiment](../opendd_support_admission_v1/README.md).
The supplement reports both the gain from learned utility ranking
and the loss in recording-level support, alongside the inconclusive
guarded comparison. No independent evidence was added.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v95
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v95.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v95.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-support-admission
```

Local and clean builds matched for all 8 main and 26 supplementary
pages at 85 DPI with the same separately supplied CVPR style. The
changed main and supplementary pages were visually checked. A clean
anonymous replay verified 106 count tables and 123 intervals across
15 readouts. This reproduces summaries, not upstream training.

The [official CVPR dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
checked November 8 are November 10 (registration), November 16 (full
paper), and November 23 (supplement), AoE. These are planning dates,
not evidence of enrollment or submission. The research is not
submission-ready; independent confirmation remains sealed.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `c6b994953c2f8b5e7b63bff2d71fdf82a93018fa90077b4561f80e70c7b08566` |
| `supplement/supplement.pdf` | `85e99aabade91354dfabf33e91df58bf2a8ec54a852dd1e8ed16cc9943413d9a` |
| `manuscript_source.zip` | `42f14ec7dbf32c29d82278c2e2e5478c853b8b5949ff8daa5a4c94f2a1b7ee14` |
| `supplement_source.zip` | `8544140ca88be6920ba28cf2858d26ee7ab99f41bf66b5a8cdbf2b8d543f85c3` |
| `opendd_timing_replay_v8/anonymous_opendd_timing.zip` | `51591c48ec4c8197401ffb8856ffa82858c5738e6d731b0b4e6ea8f039fb9bcd` |
