# AV2: selection-conditioned training does not fix position release

The [source-excluded score study](../av2_projection_risk_v1/README.md)
trained on every legal donor projection but scored one fixed projection per
view. I isolated one part of that mismatch: training on a single causally
selected projection rather than a single uniformly sampled projection.

## Controlled comparison

The two new arms use the same donor queries, direction checkpoints,
architecture, initial weights, city-balanced sampling and 400 updates.
Both have one example per donor query. The causal arm uses the original
ordered recipients chosen by the nested action model before projection;
the uniform arm uses a deterministic outcome-independent assignment
keyed by query and seed. Both donor producers exclude the source being
scored, plus the outer calibration and held cities. The causal arm is a
matched donor proxy, not the exact four-donor deployment selector.

The original transferred cost and all-legal trained score are unchanged.
Every arm sees the same fixed projected path at evaluation: same 8
observed and 12 predicted native steps, same two recipients and per-step
radii. The calibration threshold is the original largest nonempty
all-qualified prefix under the original 2% and two-floor rules. The held
readout contains 340 development scenarios, 1,020 seed views, and 119
forced passes. No new direction fit, predictor update or held selection.

| Donor training | Options across folds | Edited-easy support | Support and risk pass | Fully qualifies |
|---|---:|---:|---:|---:|
| All legal | 265,296 | 116,286 | 22,007 | 13,537 |
| Uniform one | 4,776 | 2,101 | 378 | 243 |
| Causal selected | 4,776 | 1,169 | 400 | 255 |

The causal subset has support in 24.48% of samples vs 43.99% for uniform.
Conditional on support, more causal selections pass risk (34.22% vs
17.99%), but the smaller support pool leaves the overall qualification
rate similar (5.34% vs 5.09%). This documents the composition shift; it
does not prove that support alone causes the held ranking failure.

## Held results

| Score | Same-count top five qualifies / 90 | Same-count risk fails / 90 | Mean fold AP | Calibration passes | Held releases / qualifies |
|---|---:|---:|---:|---:|---:|
| Transferred cost | 31 | 45 | 0.443 | 6 | 6 / 3 |
| All legal | 15 | 65 | 0.224 | 5 | 5 / 0 |
| Uniform one | 11 | 77 | 0.191 | 4 | 6 / 0 |
| Causal selected | 11 | 76 | 0.185 | 0 | 0 / 0 |

The causal-minus-uniform qualification contrast is 0.00 percentage
points (paired city 95% CI [-6.67, +8.89]). Against all legal,
causal selection adds 12.22 pp in risk failures ([+6.67, +17.78]);
against transferred cost it loses 22.22 pp of qualification
([-35.56, -8.89]). Same-count ranks are a diagnostic, not a release rule.

The uniform arm releases six in one city; all fail qualification and five
fail risk, despite mean conservative gains of +7.813/+31.365 pp on those
same views against CV/OLS4. It has ADE/FDE 0.22920/0.49142 and worst
released p95 ADE 2.153 in native coordinates. Its mean conservative gain
over the full CV-fallback policy is +0.0266 pp ([0.0000, +0.0797]).
Accuracy on a small subset does not establish risk transfer. An empty
causal release has no estimated released accuracy or risk rate.

Keep the frozen position projection as a development reference; do not
deploy these scores. This rules out this specific selected-sample
adjustment at the tested count. It does not resolve differences between
three- and four-donor producers or justify opening independent outcomes.
The next evidence should address edited-easy support explicitly rather
than discard more options or relax the original calibration rule.

## Reproduce or check offline

The 36 score fits (seeds 17, 29, 43 in six rotations) took 46.07 s at
742 MB peak RSS on native arm64, four Torch threads and no data workers.
Atomic checkpoints every 50 updates retain model, optimizer and RNG.
The heartbeat records PID, fold and progress. Frozen predictions and
score receipts are checked before reading held scalar outcomes. The
archive contains no raw labels, identifiers, or checkpoints.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_selected_projection_risk.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_selected_projection_risk.py score
.venv-pytorch/bin/python scripts/package_m3w_av2_selected_projection.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

The local anonymous ZIP is named `av2_selected_projection_evidence.zip`
under the experiment submission directory. From the extracted directory,
`python -I -B -S verify.py` checks hashes, reproduces the counts and
enumerates all 46,656 paired city bootstrap draws. The exact interval
for selected minus uniform qualification is [-7.78, +8.89] pp. This
replays scalar evidence, not training or independent confirmation.

[Freeze](freeze.json), [calibration](calibration.json), [metrics](summary.json),
[offline replay receipt](anonymous_replay.json).

The existing v166 main PDF/source and v165 supplement PDF/source still
match their recorded hashes (8 and 50 PDF pages, respectively). The main
source uses review mode, lists an anonymous author and places the LLM
statement before references. These are file checks, not a claim that
submission is ready. The [official schedule](https://cvpr.thecvf.com/Conferences/2027/Dates)
still lists November 16 AoE for the full paper and November 23 AoE for
supplementary material. See the [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
for page, anonymity and disclosure rules. No manuscript or submission
change, no official validation/test, DroneCrowd confirmation, new
CREATE job, deployment, Stage5C or SMC.
