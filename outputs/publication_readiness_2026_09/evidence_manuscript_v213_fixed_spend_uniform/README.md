# Manuscript v213: add the equal-budget uniform control

The OpenDD results now compare frozen scores against a uniform legal
allocation under the same total edits, chronological credits and
movement cap. Uniform gives 49.29 expected complete-grid passes
versus 55 for the frozen scores. The equal-location difference is
-0.37 pp [95% interval -1.82, +1.07]. The 210/606 hindsight result
is unchanged. The new control uses whole-recording eligibility, so
it is not presented as an online policy. No external result or
release decision changed.

The eight-page anonymous main PDF and its source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v213_fixed_spend_uniform/`.
The title, abstract and LLM Usage Statement are unchanged. The paired
`review_supplement_v47.zip` under `submission_admin/` replays the new
expectation and intervals along with the inherited evidence; its
reader PDF remains v44. No document was uploaded or submitted.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_fixed_spend_uniform.py
python scripts/build_m3w_fixed_spend_uniform_candidate.py --compiler "$TECTONIC"
```

The build uses verified v212 artifacts and cached Tectonic inputs;
it needs NumPy, pypdf and ReportLab. The anonymous archive requires
only standard-library Python. The archive reproduces aggregate
arithmetic, not inference or the original uniform sampling.

[Comparison](../opendd_fixed_spend_uniform_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json) |
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
