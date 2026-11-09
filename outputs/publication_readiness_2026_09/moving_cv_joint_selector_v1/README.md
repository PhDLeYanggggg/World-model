# Causal Pair Coupling at Matched Edit Counts

I tested whether the additional choices in the frozen neural forecast bank
can be selected using only past information. I kept the same CV fallback,
seven alternatives, eight-edit cap and realized edit count in each query.
The independent, uniform-family, additive-pair and coupled selectors all
made 12,896 edits per seed, 38,688 per arm over three seeds. The coupled and
additive arms share the same row and edge models and the same forecast for
any recipient they share. Only the learned nonadditive pair term differs.

| Coupled contrast | Agent observed gain (pp) | Agent full-grid lower gain (pp, 95% CI) | Pair observed gain (px) | Pair full-grid lower gain (px, 95% CI) |
|---|---:|---:|---:|---:|
| vs. independent | -0.001 | -1.680 [-3.161, -0.573] | +0.282 | -1.536 [-3.518, -0.269] |
| vs. additive | -0.021 | -1.117 [-2.426, -0.220] | +0.209 | -1.145 [-2.920, -0.109] |
| vs. additive, equal total peak per query | +0.002 | -1.173 [-2.594, -0.228] | +0.222 | -1.154 [-2.940, -0.113] |
| vs. uniform family | -0.131 | -4.117 [-7.945, -1.148] | +0.106 | -2.997 [-6.378, -0.766] |

The +0.209 observed pair effect against additive has a 95% interval of
[+0.012, +0.536] pixels. It does not carry over to the full-grid paired
lower bound. The coupling changed 2,345 recipients, so the negative
conservative result is not explained by an inactive decoder. Only 3/14
sources have positive agent lower gain against additive, and only 2/14
after matching total peak displacement. Unknown outcomes are not treated
as zero error. The original 2% rule passes **2/102** views for each forced
selector; worst defined signed easy degradation is +265.07% for the raw
joint arm. Leaving CV unchanged is a fallback, not a supported edited policy.

The joint arm's full-grid lower gain is -7.641 pp versus CV and -31.778 pp
versus OLS4. On available labels alone, its source-macro ADE/FDE is
40.115/107.394 pixels, versus 40.549/108.777 for CV and 33.885/92.570 for
raw EqMotion. These are different summaries from the conservative scores;
the endpoints are diagnostic when only a subset is labeled.

## Interpretation

The previous hindsight result proved that the expanded menu contains
additional feasible combinations. It did not promise that the learned
selector could find them. This experiment provides evidence that a
nonadditive pair head can improve labeled pair geometry, but not that it
improves conservative utility or makes edits admissible. Equalizing the
total displacement per query does not remove the deficit. In a separate
past-only audit, the donor/A source-averaged 99th-percentile normalized
EqMotion edit peaks are 20.77/11.80; for PCT they are 19.56/12.52. Producer
maturity and source motion both differ, so the shift is not a causal
explanation of the failure. The next useful check is to learn admission
from support and paired gain on nested fitting sources, and test it against
the same simple ranking control before opening any independent outcomes.

Learning to defer among experts and decision-focused ranking have direct
precedents ([Mao et al.](https://proceedings.mlr.press/v235/mao24d.html),
[Mandi et al.](https://proceedings.mlr.press/v162/mandi22a.html)). The
comparison here isolates the causal pair term at fixed query counts and
separates observed pair improvement from recording risk and missing labels.

## Reproduction and scope

`fresh_run`: 18 source-excluded donor fits (10,800 updates) and six choice
heads (7,200 updates), plus frozen A inference and readout. The heads and
readout took 48.03 s locally at 3.59 GB maximum resident memory.
`cached_verified`: all donor checkpoints and forecast identities, 2,304
exact donor prediction rows, zero-update replay of the full selector, and
anonymous scalar replay. 33 focused tests passed. Bootstrap: 3,000 source
draws after averaging three seeds over 34 exposed recordings / 14 sources.
`not_run`: independent confirmation, new calibration, deployment, Stage5C
or SMC. The 30 DroneCrowd confirmation groups remain sealed. Native eight
observed and 12 future annotation steps, local image pixels, not meters,
seconds, physical collision safety, true 3D, or a foundation model result.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_joint_selector.py
.venv-pytorch/bin/python scripts/run_m3w_cv_joint_selector.py --replay
.venv-pytorch/bin/python scripts/analyze_m3w_cv_joint_selector.py
.venv-pytorch/bin/python scripts/build_m3w_cv_joint_selector_replay.py
.venv-pytorch/bin/python scripts/plot_m3w_cv_joint_selector.py \
  outputs/publication_readiness_2026_09/moving_cv_joint_selector_v1/readout.json \
  data/stage_cvpr2027_experiments/moving_cv_joint_selector_v1/paired_contrasts.png
```

The anonymous replay was extracted and verified outside the repository;
archive SHA-256: `9245c151f490711b58fc322bb10f4794ef842470bd447812e776e57042eca2a1`.

[Donor bank](../moving_cv_joint_oof_v1/README.md) | [Protocol](protocol.md) |
[Readout](readout.json) | [Source diagnostic](source_diagnostic.json) |
[Manuscript v125](../evidence_manuscript_v125/README.md)
