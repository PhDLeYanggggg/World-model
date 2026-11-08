# Evidence Manuscript v90

I added the [recording teacher and support study](../opendd_record_teacher_v1/README.md).
The matched teacher result is negative for lower utility. The separate
extra-edit analysis changes the diagnostic budget, retains the original
risk rule, and uses outcomes. It is not a new deployment result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v90
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v90.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v90.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-record-teacher
```

Local and clean builds use the same separately supplied CVPR style.
All 8 main and 25 supplementary pages matched at 85 DPI; main page 7
and supplement pages 24 and 25 were visually checked. The anonymous
timing archive reproduces 40 count tables and 40 intervals in five
studies from anonymous scalars. It does not reproduce upstream training.
Generated PDFs, ZIPs and checkpoints remain local. No submission was made.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `51c2a3ad6a5f1a2e542968ee673f6cb93d39e52acdfae5b428bd326865cb7749` |
| `supplement/supplement.pdf` | `d693d6f010bbea1e0545a8722b9e841ebb4c3c623cf61613d2869f051c84f697` |
| `manuscript_source.zip` | `1fbdf67cc812e17d999b524c75bbbbb7edb50cd83889cdaccf2e4dbb4d8b6e7f` |
| `supplement_source.zip` | `2620fec1749da2926e709ceb244e298db5e7ab0cddbc14c20988851657a87047` |
| `opendd_timing_replay_v3/anonymous_opendd_timing.zip` | `65deb169d3c8061a9fb6a5cffcb5569195e03f36e723ac0e4775469944f1a208` |
