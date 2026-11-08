# Working Manuscript v55: Observed-Window Replay

Section 38 adds the causal short-prefix experiment and the exact
same-count permutation control. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. The control does not support an individualized assignment
claim. The recordings remain development data.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v55
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v55.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render identical fifteen-page images.
The new section and references were visually checked. A fresh scratch
replay and a repeated permutation audit produced the original results.
No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `89d0be8ab0a30f328ba80bab44b048ee966ac22b7e342add1389aa24cae994f6` |
| Supplement source ZIP | `9235c8cbb39d6d86fb24a5b0e733fb36820221ba61f2aa9225aba0473901d18a` |

[Experiment and reproduction](../prefix_path_replay_v1/README.md).
