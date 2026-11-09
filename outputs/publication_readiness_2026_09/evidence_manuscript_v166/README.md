# Working manuscript v166

The abstract and opening now lead with the comparison: fix edit count
and displacement, then ask what a past-only policy can recover from
the forecasts that are already available. The revision keeps the
OpenDD development counts and the frozen WEPDTOF result distinct.
It updates the related-work discussion and describes how Codex was
used in the research and writing. The figures, experimental values,
and [v165 supplement](../evidence_manuscript_v165/README.md) are unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v166
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v166.py --source-zip "$out/main_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `3541d2c7cc29b04ea8db3c74151a6850530d43ac4a071f3e8dc7add15210e6ad` |
| `main_source.zip` | `6a43eb501acf5c12ece358339e2c6f9f9c2c904a4117af9a7acabb0cbeb92532` |

All eight pages match a clean compile from the archive at 90 DPI, with
the CVPR style supplied separately. The manuscript is a working draft;
the PDF and source archive stay local. The build does not submit anything.
