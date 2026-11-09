# Main manuscript v167: dataset citation

I added the [WEPDTOF benchmark citation](https://openaccess.thecvf.com/content/WACV2022/html/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.html)
where the paper introduces its frozen external result. The bibliography
uses the reference font size from the official CVPR 2027 template.
No result, figure, abstract, title, or LLM disclosure changed. The
original v166 manuscript and both supplementary PDFs stay intact.

| Local artifact | Path under `data/stage_cvpr2027_experiments/` |
| --- | --- |
| New eight-page PDF | `evidence_manuscript_v167_citation/manuscript.pdf` |
| New source archive | `evidence_manuscript_v167_citation/main_source.zip` |
| Five-page reviewer companion | `submission_admin/compact_review_supplement_v1.pdf` |
| Technical appendix | `evidence_manuscript_v165/supplement/supplement.pdf` |
| Anonymous statistical replay | `submission_admin/reviewer_evidence.zip` |

The local author-kit archive matched the copy served by the
[official author guide](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines).
Two offline builds produced identical PDFs and source ZIPs. Pages 1
through 6 match v166 pixel for pixel at 95 DPI; pages 7 and 8 were
rendered and inspected. All 14 bibliography entries have in-text
citations; no citation is undefined. The compiled PDF has no personal
author metadata or outbound links. Fourteen focused tests passed.

```sh
# Use a Python environment with pypdf and a locally cached Tectonic bundle.
python scripts/build_m3w_citation_candidate.py --compiler "$(command -v tectonic)" \
  --receipt outputs/publication_readiness_2026_09/evidence_manuscript_v167_citation/receipt.json
python -m unittest discover -s tests -p 'test_m3w_citation_candidate.py'
```

The [receipt](receipt.json) binds the original and revised artifacts.
This is a typeset candidate, not a new evaluation, approval of the
submission, or an OpenReview upload. It contains no Songdo outcome.
