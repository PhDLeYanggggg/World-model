# Anonymous supplementary bundle

I combined the existing 5-page reading guide, 50-page technical appendix,
and two offline replays. The guide's original `reviewer_evidence.zip`
reference resolves inside the bundle. The second replay independently
recomputes the MovingDroneCrowd++ peak-budget bound. All PDFs are byte-for-byte
unchanged; the eight-page main PDF stays separate.

| Local file under `data/stage_cvpr2027_experiments/` | Use |
| --- | --- |
| `evidence_manuscript_v167_citation/manuscript.pdf` | Main PDF |
| `submission_admin/review_supplement_v1.zip` | Combined supplementary bundle |
| `submission_admin/compact_review_supplement_v1.pdf` | Short guide if a PDF alone is needed |

The bundle is about 1.33 MB. It carries no raw data files, model
checkpoints, author records or repository links. The appendix's one live
external link is the citation to the published MANTRA paper. The official
2027 LaTeX kit omits a paper ID from the rendered PDF.

## Verify

```sh
# Use a Python environment with pypdf installed for the build.
python scripts/package_m3w_combined_review_supplement.py
# Extract the resulting ZIP, then run from its root (standard library only).
python -I -B -S verify.py
```

The isolated replay verifies OpenDD 330/606 hindsight witnesses and 57/606
past-only passes, MovingDroneCrowd++ 67/72 OLS4 exclusions, and WEPDTOF 0/45
fixed-joint dual passes. All 18 focused checks passed across the bundle,
main citation and two original replay suites. [Hash and result receipt](verification.json).
No outcome was opened, no paper text was changed, and no file was uploaded.
