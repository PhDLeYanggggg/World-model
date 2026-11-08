# Exact Query-Matched Hindcast Replacements

The [peer models](../tumdot_peer_choice_v1/README.md) all change
the same rows versus CV at the same displacement radii, but they
do not change away from hindcast equally often. On each current
query, keep the minimum of the three original guarded replacement
counts. Each model keeps its own top predicted-gain replacements;
all other rows use hindcast. Matching uses only the frozen causal
scores and current paths, never observed outcomes or future masks.
This is a shared-budget comparison, not a deployable policy.

| Half-radius matched arm | Replacements | Lower vs CV (pp) | Risk / 27 | Joint CV and OLS4 / 27 |
| --- | ---: | ---: | ---: | ---: |
| Blind | 1,091 | -0.00640 | 24 | 9 |
| Aligned | 1,091 | -0.00511 | 24 | 8 |
| Permuted | 1,091 | -0.00800 | 24 | 8 |

Aligned minus blind is +0.00244 pp on observed steps but
**-0.01497 pp [-0.02706, -0.00618]** after exact pair disagreement.
Against permuted, the corresponding values are +0.00191 and
**-0.01677 pp [-0.03306, -0.00092]**. All three sites are negative
in both direct lower contrasts. Matching reduces the gap relative
to the original comparison but does not remove it. The
three-intersection intervals are descriptive, not confirmation.

Quarter radius matches 1,088 replacements per arm and passes the
original risk rule in 27/27 views, but the same pair contrasts
remain negative (-0.00560 and -0.00853 pp). No recorded learned
arm beats CV on average at either radius. No learning or threshold
search occurs in this audit. Original 2% rule, protected confirmation
groups and existing policy remain unchanged. Stage5C and SMC are off.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_peer_budget.py
.venv-pytorch/bin/python scripts/build_m3w_tumdot_peer_replay.py
```

The audit reproduces the original unthinned contrast and count
before scoring each view. Repeating the command checks the same
write-once outputs. See [registration](registration.json) and
[scalar results](analysis.json).
