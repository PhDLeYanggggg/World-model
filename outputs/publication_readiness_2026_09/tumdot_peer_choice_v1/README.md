# Does Aligned Peer Context Improve TUMDOT Choice?

I keep the [matched TUMDOT roster](../tumdot_joint_teacher_v1/README.md):
nine previously exposed recordings, 404 nonoverlapping queries,
1,456 targets, six directions, three seeds and one campaign. The
registry's released files reconstruct every history row and query
offset exactly. Both the held intersection and its collection day
are excluded from fitting. The acquisition campaign and upstream
trajectory release are not independently held out.

The aligned model sees nearest-peer separation, relative motion
and candidate spacing against the current peers' fixed causal
hindcasts. The blind model zeros these columns. The permuted model
cyclically reassigns the columns within the same current query,
preserving the feature marginals while breaking actor alignment.
No later frame or query is used for the permutation. All three
heads use the same paired gain/harm labels, architecture, draws
and initialization. Held future labels are used only for scoring.
This tests contextual inputs to a row-wise choice, not simultaneous
online joint optimization or physical safety.

| Half-radius guarded arm | Lower vs CV (pp) | Risk / 27 | Joint CV and OLS4 / 27 | Hindcast replacements |
| --- | ---: | ---: | ---: | ---: |
| Fixed hindcast | -0.03654 | 24 | 3 | 0 |
| No peer | -0.01054 | 24 | 9 | 1,425 |
| Aligned peer | -0.00871 | 24 | 7 | 1,827 |
| Permuted peer | -0.01181 | 24 | 6 | 2,086 |

Direct paired conservative utility for aligned minus blind is
-0.04168 pp [-0.07135, -0.00177]; aligned minus permuted is
-0.05478 pp [-0.05716, -0.05280]. The small improvement against
CV is not a direct win over those alternative paths. The three
intersection resamples are descriptive. No learned arm achieves
positive mean lower utility against CV at half radius.

With 81,207 available pair-steps across repeated half-radius views,
observed relative-displacement ADE for the guarded aligned arm is
0.880674 in native coordinates versus 0.880658 for blind. This
does not establish an interaction benefit or collision safety.
The [count-matched diagnostic](../tumdot_peer_budget_v1/README.md)
checks whether different replacement frequency explains the loss.

**Decision:** no promotion. This configuration does not justify
a larger neighbor encoder on the same exposed campaign. The
next useful evidence is an additional acquisition group with
verified input and reference support, before investing in another
TUMDOT-only feature change. Existing [source intake](../station_sharedspace_intake_v1/README.md)
identifies Hannover as a distinct single-group check, not a new
official confirmation split. The original 2% rule and sealed
DroneCrowd groups are unchanged.

```sh
.venv-pytorch/bin/python scripts/run_m3w_tumdot_peer_choice.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_peer_choice.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_peer_budget.py
.venv-pytorch/bin/python scripts/build_m3w_tumdot_peer_replay.py
```

`fresh_run`: 81 fits, 32,400 updates, 25.638 s, PID 92644. Native
arm64 Torch, four threads, zero workers, checkpoints/heartbeat
every 100 steps. Rerun to resume. `cached_verified`: zero-update
replay, publisher files and fixed matched paths. `not_run`: CREATE,
deployment, Stage5C, SMC, or independent confirmation.
