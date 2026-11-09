# Five-page review supplement

I made a short English reading companion to the eight-page working
paper. It puts the comparisons a reviewer needs to check in one place:

| Page | Content |
| --- | --- |
| S1 | Main claim, controls, and the three evaluation rosters |
| S2 | The original 2% recording rule, missing-label charges, and source intervals |
| S3 | Frozen WEPDTOF comparison, no-edit cases, and matched source contrasts |
| S4 | OpenDD vehicle headroom and the equal-capacity anchored ranking comparison |
| S5 | Same-cohort vehicle forecast controls and an offline replay command |

The PDF does not replace the 50-page technical appendix, which contains
analyses cited elsewhere in the paper. The main paper and original
appendix were not edited. The short version cites the
[WEPDTOF benchmark paper](https://openaccess.thecvf.com/content/WACV2022/html/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.html);
the existing main bibliography does not yet include that citation.

## Local files

| Artifact | Location under `data/stage_cvpr2027_experiments/` |
| --- | --- |
| English main manuscript | `evidence_manuscript_v166/build/manuscript.pdf` |
| Short review supplement | `submission_admin/compact_review_supplement_v1.pdf` |
| Original technical appendix | `evidence_manuscript_v165/supplement/supplement.pdf` |
| Anonymous replay | `submission_admin/reviewer_evidence.zip` |

The PDF has five section bookmarks, no outbound links and no author
name in the metadata. All pages were rendered and inspected. Two
builds matched byte for byte; extraction checks confirm the headline
counts and paired contrasts are on the expected pages. The archive
replayed in a clean directory without using the repository, data
or model weights. It checks saved statistics, not raw tracking or
training. Two PDF tests and twelve evidence replay tests passed.

```sh
# Use a Python environment with ReportLab and pypdf.
python scripts/build_m3w_compact_supplement.py --receipt outputs/publication_readiness_2026_09/compact_review_supplement_v1/receipt.json
python -m unittest discover -s tests -p 'test_m3w_compact_supplement.py'
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_main_claims_replay.py tests/test_m3w_reviewer_evidence.py tests/test_m3w_wepdtof_scalar_replay.py
```

The [build receipt](receipt.json) binds the local PDF, main paper,
replay archive and source readouts. `fresh_run` refers to the PDF
build and offline replay; the underlying experiments are
`cached_verified`. No new outcomes were opened, no results were
retuned, and nothing was uploaded to OpenReview.

## Remaining decisions

1. The independent Songdo comparison is still the most useful new
   experiment. Its protocol choices are already rehearsed; no real
   outcome readout is included here.
2. The 8-page manuscript can use a direct WEPDTOF citation. The
   [CVPR author instructions](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
   also call for accurate ethics/consent reporting where applicable.
   Existing PDF text does not document an institutional determination.
   Public access and publisher terms are not a determination of
   consent or exemption. No institutional claim is inserted.
3. Use the [official dates](https://cvpr.thecvf.com/Conferences/2027/Dates):
   full paper November 16 and supplement November 23, AoE.
   This artifact does not establish portal or submission status.
