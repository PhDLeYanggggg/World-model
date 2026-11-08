# Working Manuscript v60: Causal Reference Bridges

Section 43 reports the learned policy, the exact same-cap ceiling
and the separate gain/risk counterfactuals. The
[v50 main paper](../evidence_manuscript_v50/README.md) is unchanged.
The results do not establish a new deployment policy.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v60
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v60.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render identical sixteen-page
images. The new section and references were visually checked. The
held-location score replayed with zero updates, and the matched-cap
diagnosis reproduced exactly. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `10230c45d0f64be4062d15f4c4c28ce3b1780edd54348d9be2fdc58d23dd7847` |
| Supplement source ZIP | `ea01e3f93f1fe53571680966a0c574f4f13368a532c27f03d1c3c8cef04e8a12` |

[Experiment and reproduction](../reference_bridge_policy_v1/README.md).
