# Paired-Loss Gradient Audit

All nine frozen models, complete own-B fitting populations and exact
saved query-frequency weights. No new V/H readout or saved model update.

| Context | Arm | Risk/rank0 cosine | Risk/rank1999 cosine | Base/rank1999 cosine | Extra-risk norm / anchor |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | joint | -0.86658 | -0.863172 | 0.449283 | 1.48883 |
| eqmotion_fold1_seed17_critic0 | paired_point | -0.99556 | -0.996019 | -0.320571 | 0.227984 |
| eqmotion_fold1_seed17_critic0 | paired_rank | -0.927862 | -0.925437 | -0.933634 | 0.340189 |
| eqmotion_fold0_seed17_critic1 | joint | -0.121523 | -0.119122 | 0.0552101 | 0.228634 |
| eqmotion_fold0_seed17_critic1 | paired_point | -0.969629 | -0.968506 | 0.209882 | 0.22185 |
| eqmotion_fold0_seed17_critic1 | paired_rank | 0.0776027 | 0.148454 | -0.32781 | 0.255831 |
| eqmotion_fold0_seed17_critic2 | joint | -0.710004 | -0.709587 | -0.0595058 | 0.0392169 |
| eqmotion_fold0_seed17_critic2 | paired_point | -0.741626 | -0.743965 | 0.477994 | 0.0857332 |
| eqmotion_fold0_seed17_critic2 | paired_rank | -0.679455 | -0.697222 | -0.314574 | 0.0374657 |

## Disposable Saved-AdamW Steps

| Context | Variant | Mean risk MSE | Joint decision cost | False-safe | False-reject |
|---|---|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | base | 0.00974750862 | 0.00716583106 | 4 | 2921 |
| eqmotion_fold1_seed17_critic0 | rank | 0.00978254548 | 0.00702483309 | 4 | 2880 |
| eqmotion_fold1_seed17_critic0 | risk_detached_rank | 0.00974750862 | 0.00721270911 | 4 | 2942 |
| eqmotion_fold1_seed17_critic0 | risk_protected_rank | 0.00974223386 | 0.00721167984 | 4 | 2936 |
| eqmotion_fold0_seed17_critic1 | base | 0.00841116593 | 0.00301262188 | 43 | 1870 |
| eqmotion_fold0_seed17_critic1 | rank | 0.00841094736 | 0.00298227623 | 43 | 1870 |
| eqmotion_fold0_seed17_critic1 | risk_detached_rank | 0.00841116593 | 0.00298227623 | 43 | 1870 |
| eqmotion_fold0_seed17_critic1 | risk_protected_rank | 0.00841094736 | 0.00298227623 | 43 | 1870 |
| eqmotion_fold0_seed17_critic2 | base | 0.612920664 | 0.0195336928 | 169 | 22572 |
| eqmotion_fold0_seed17_critic2 | rank | 0.612924467 | 0.0195318516 | 169 | 22571 |
| eqmotion_fold0_seed17_critic2 | risk_detached_rank | 0.612920664 | 0.0195318516 | 169 | 22571 |
| eqmotion_fold0_seed17_critic2 | risk_protected_rank | 0.612920098 | 0.0195318516 | 169 | 22571 |

Risk/paired conflict at both offsets: 2/3.
Protected step lowers MSE without higher joint cost: 1/3.
Prospective training follow-up supported: **False**.

These are final-model local derivatives and disposable full-fitting steps,
not the full training trajectory, held generalization or an empirical safety
certificate. AdamW momentum and clipping are retained. Unknown-label bounds,
original 2% rules and support remain unchanged. No identity exemption.
A negative result rejects this particular conflict repair, not all models.

Original 24-group primary remains NA; independent confirmation not_run.
Native obs8/pred12 image coordinates, not metric/seconds/true3D/foundation/
human gold. Stage5C and SMC remain disabled.

[Protocol](protocol.md) | [Metrics](analysis.json)
