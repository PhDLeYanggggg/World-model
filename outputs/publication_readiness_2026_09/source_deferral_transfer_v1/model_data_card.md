# Frozen Deferral Models and Source Data Card

## Model

Six frozen Torch models: two cost objectives, seeds 17/29/43, final update
10,000, 44,897 parameters each. Each adds an observed cost score to a bounded
twelve-step proposal. Proposal use requires score > 0; otherwise exact
stationary CV. The mask-only geometry/coverage input is unchanged from matched
dense models (44,864 parameters). No architecture or weight is changed by this
readout. RGB is explicitly zeroed in this control arm, so this is not evidence
for a useful multimodal contribution.

Training ancestry comprises three verified 2,000-update parents and six
8,000-update continuations. That training was completed under a separate
registration; this experiment contributes **zero training updates**. Pure
expected-cost and signed-cost-supervised objectives are described in the
[training card](../source_cost_deferral_v1/model_data_card.md). The additional
supervision/proposal-loss package is not an isolated loss ablation.

## Data and Roles

Original admitted SDD training recordings only. The current stationary-history
source subset has 15,430 fit and 6,944 diagnostic rows. Diagnostic rows span
seven recordings and 181 recording-scoped agents at bookstore; this site was
not fitted by these models but has prior exploratory exposure. Fit rows span
four sites, 29 recordings and 545 scoped agents. Neither subset represents the
full SDD benchmark or a pristine test population.

Eight observed annotation positions and twelve targets at source stride 12;
forecast offset +144 raw annotation frames. Supplied historical annotations can
be interpolated from later annotation controls. This is offline supplied-history
forecasting, not verified sensor-as-of data. Explicit future labels are only
supervision/evaluation; they do not enter observed features, neighbor selection
or goals. No test endpoints construct goals and no central velocity is used.

Normalization and hard cutoff use training or past-only information under the
fixed contract. Input caches, row identities and prediction alignment are
hash-bound. This scoped audit does not retroactively clear the older Stage37
lineage/selection issues. Main selection, calibration and confirmation roles
were not scored.

## Results and Use Limits

Best-looking cost-supervised hard-action aggregate is still -1.245558% versus
CV, with all three seeds negative. Zero-target harm is positive and percentage
degradation undefined. No model is selected or promoted. A learned score is not
a calibrated failure probability or a physical safety guarantee.

Intended use: reproducible source diagnostics and training-side hypothesis
tests. Not intended for deployment, physical navigation or publication claims
of world-model success. Source units are annotation pixels/past-normalized
coordinates and raw frames; no meter/second, true-3D, foundation or human-gold
claim. Stage5C and SMC remain disabled.

## Reproducibility and Distribution

Native arm64 `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, CPU four threads,
interop one, no DataLoader subprocesses. All nine checkpoint outputs replay
exactly. Aggregate results, code and original summary SVG are public; source
data, row-level arrays, images, caches and checkpoints remain local and are
not included in the Git update. Public code alone cannot reproduce scores
without obtaining and constructing the referenced data/checkpoints.
