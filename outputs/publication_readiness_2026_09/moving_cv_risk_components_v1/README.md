# Separating harm, missingness, and easy support

I kept gains, candidate paths, causal features and training samples
fixed. Both new heads predict the same marginal risk moments and
three mutually exclusive support classes. The direct head fits
total harm. The component head predicts conditional observed
harm and easy benefit, then combines them with class probabilities.
Its expected wholly missing debit is the probability of that
class times the candidate's exact peak displacement from CV.
Reference mass and class probability depend only on history,
not which path is proposed.

This is a conditional-expectation construction, related to
[two-part models](https://www.tandfonline.com/doi/abs/10.1080/07350015.1983.10509330).
It does not assume independence between easy status and harm.
Partial missingness still enters conservative utility through
the unchanged step-level comparison, not this whole-row risk debit.
Both heads use identical marginal squared losses and class loss.

## Fixed-count and free policies

18 local arm64 PyTorch fits, 25,200 updates, three seeds and three
source rotations. Held development contains 24 recordings in
nine source groups, or 72 recording/seed views.

| Policy | Edits | Original risk passes / 72 | Undefined easy / 72 | Worst known signed easy |
|---|---:|---:|---:|---:|
| Frozen shape portfolio | 45,042 | 4 | 14 | +10.21% |
| Shape risk, independent matched | 45,042 | 3 | 17 | +25.85% |
| Direct risk, independent matched | 45,042 | 3 | 19 | +30.25% |
| Component risk, independent matched | 45,042 | 3 | 19 | +40.55% |
| Direct risk, joint free counts | 13,430 | 10 | 26 | +12.83% |
| Component risk, joint free counts | 21,242 | 20 | 29 | +12.48% |

| Contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Components vs direct, exact high counts | +0.008 | -0.515 [-0.860, -0.211] |
| Components vs shape independent, exact high counts | -0.046 | -0.870 [-1.327, -0.435] |
| Components vs direct, joint free counts | -0.082 | -0.554 [-1.286, -0.040] |
| Matched components vs CV | +0.626 | -0.194 [-0.822, +0.335] |
| Matched components vs OLS4 | +0.025 | -54.537 [-96.379, -22.360] |

Intervals use 3,000 paired source resamples after averaging seeds.
The matched pair-distance effect is -0.738 local pixels against
direct risk [-1.181, -0.380]. Calibration qualifies 0/9. The free
policy pass difference is not a comparison at the same counts.

## What the controls show

The low-count diagnostic fixes the joint component policy's exact
query counts for all three arms (21,242 edits, 10.15% of rows).
Independent direct/component and joint component pass **2/1/20**
of 72 original checks. All three have 29 views with undefined
easy support. Joint vs component independent improves source-balanced
pass rate by +14.42 pp [0.00, +31.75] but loses 0.507 conservative
accuracy pp [-0.831, -0.209]; its worst signed easy change is
still +12.48%. At the same low counts, components lose 0.259
conservative pp [-0.494, -0.083] to direct scoring. This control
shows a risk-utility tradeoff conditional on the joint policy's
selected schedule, not a general safety improvement.

On the original shape portfolio's common exposures, the component
head's easy-class Brier score is worse: direct minus component
is -0.0307 [-0.0553, -0.0075]. The difference in missing-debit MAE
is +0.0046 [-0.0010, +0.0142]; the interval includes zero.
Replacing only positive-harm estimates in direct selection loses
0.465 conservative pp [-0.748, -0.214]. Replacing signed change
or reference estimates alone also fails (-0.213 and -0.096 pp).
These substitutions change the predicted decision objective;
they are not new calibrated policies.

**Next:** retain the joint constraint as an experimental arm, not
the component head as a winner. Learn utility against OLS4
within low-risk alternatives, then compare at the same risk and
support budget, alongside the exact-count control. The current
joint method buys more risk passes with less conservative accuracy;
the question is whether another objective can move that tradeoff.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_risk_components.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_component_swaps.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_risk_component_low_count.py
```

Training, scoring and both controls are `fresh_run`. Shape
checkpoints and the original portfolio are `cached_verified`.
Replay reproduces the frozen readout with no optimizer updates.
Omit `--replay` to train or resume. Fresh runtime was 87 seconds
with four compute threads and no DataLoader workers. No CREATE
job was needed. Native eight observed / twelve forecast steps in
dataset-local pixels, not metric or seconds. No deployment,
Stage5C, SMC, or submission. DroneCrowd's 30 independent groups
remain sealed under the approved 15/30 split.

[Protocol](protocol.md) | [Main readout](readout.json) |
[Block swaps](block_swap_readout.json) |
[Low counts](low_count_readout.json) |
[Working supplement](../evidence_manuscript_v145/README.md).
