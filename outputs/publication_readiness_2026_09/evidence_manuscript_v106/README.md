# Evidence Manuscript v106

This draft adds [direct pass-transition supervision](../opendd_pass_transition_v1/README.md).
The fixed-menu model recovers more passes than the previous head
but destroys more, and loses to the count-matched composite in
conservative utility. The geographic concentration of feasible
actions is reported separately. No confirmation outcomes were opened.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v106
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v106.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v106.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-pass-transition
```

Clean source-archive builds matched all 8 main and 30 supplementary
pages at 85 DPI with the same separately supplied CVPR style.
Main page 7 and supplementary pages 29 and 30 were visually checked.
The standalone anonymous replay verified 206 count tables and 305
paired intervals across 35 readouts. These checks establish
reproducibility, not independent performance or submission readiness.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `f94aaac993a974b9d44899528f1fd534877464267dd008f241de077d7ca48b82` |
| `supplement/supplement.pdf` | `a002c7887c5981fb8bef11214d3564ea1cee10cdc7584bcab0bc88a649ec9ba7` |
| `manuscript_source.zip` | `0cce9e54497353cda86fccffebbdc83ed2b19f1b8430912a503e8347217eded4` |
| `supplement_source.zip` | `5a913520323848d69059ce0a242c943641db1e7037fbea824fa4d5d7973d64ec` |
| `opendd_timing_replay_v19/anonymous_opendd_timing.zip` | `e0b5a9b6b8cf90744ef75c454ac508b612a10484edba214b4994ea8998935918` |
