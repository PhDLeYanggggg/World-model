# M3W Paper Draft

## Title

M3W: Leakage-Free Multimodal Agent-Scene World Modeling for Top-Down Multi-Agent Forecasting

## Abstract

We study whether JEPA-style representation learning and spatiotemporal Transformer dynamics can improve real-world top-down multi-agent world modeling under strict no-leakage evaluation. Current local-small evidence does not yet establish M3W as a submission-ready method: the Stage26 cost-aware selector remains the best deployable model. We report the negative result, runtime fixes, ablations, and the remaining evidence gap.

## Introduction

The problem is to learn a multimodal 2.5D agent-scene world model that improves over strongest causal baselines and Stage26 selectors on SDD pixel-space raw-frame horizons while preserving easy cases.

## Related Work Placeholder

Trajectory forecasting, JEPA/self-supervised world representation, top-down pedestrian/drone datasets, safety-gated selectors.

## Method

M3W uses tokenized agent, scene, goal, interaction, baseline rollout, horizon, dataset, time, and mask features. JEPA predicts latent targets; Transformer dynamics aggregate tokens; downstream heads predict expected FDE, failure, goal diagnostics, interaction risk, occupancy, and physical validity.

## Datasets

SDD is used as pixel-space official benchmark. t+50 is official raw-frame; t+100 is raw-frame diagnostic. Effective seconds and metric scale are not verified.

## Metrics

t+50 FDE improvement, t+100 diagnostic, hard/failure improvement, easy degradation, AUROC/AUPRC/ECE, interaction risk, occupancy, bootstrap CI, ablations.

## Baselines

strongest causal baseline, BPSG-MA v1 fallback, Stage26 selector, JEPA-only, Transformer-only, Hybrid.

## Experiments

See `experiment_matrix.md`, `ablation_table_m3w.md`, and `bootstrap_or_seed_report.md`.

## Ablation

Current ablations are inference-time token masks. Retrained ablations are required before paper submission.

## Failure Analysis

Hybrid does not beat Stage26; JEPA non-collapse/downstream lift not proven; hard/failure gate not consistently passed by M3W.

## Limitations

Not true 3D, not metric, not foundation model, no Stage5C execution, no SMC, no human-gold scene labels.

## Reproducibility

Use `.venv-pytorch/bin/python` arm64, `num_workers=0`, checkpointed configs, and no future/test leakage.
