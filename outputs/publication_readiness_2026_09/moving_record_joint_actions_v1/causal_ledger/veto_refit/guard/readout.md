# A Calibrated Release Guard Gives Up Utility Without Solving Risk

For each of 42 source/forecast-seed cells, I separated one evaluation
source, three calibration sources, and ten fitting sources. The cost
fit, residual refit and learned harm score exclude both calibration
and evaluation sources. The inner donor proposal also excludes its
own source. The calibration sources choose only the threshold; the
outer decisions freeze before evaluation. All four arms match at
every query (6,344 switches each). This count differs from the
previous 13-source fit, so I do not compare the two as matched.

| Policy | Paired Lower vs Base (pp) | Selected Reopened Actions | Easy Net Above 2% | Easy Positive Harm Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|
| Ten-source base | 0 | 0 | 2 | 59 | 41 | 2 |
| Unguarded reopening | +0.20260 | 2,670 | 8 | 62 | 37 | 3 |
| Learned easy-harm guard | +0.05579 | 1,039 | 5 | 62 | 38 | 2 |
| Normalized radius guard | +0.08172 | 464 | 4 | 64 | 35 | 3 |

The unguarded paired lower vs base is +0.20260 pp
[+0.01963, +0.41839]. Neither guard retains a reliable whole-roster
gain: learned +0.05579 pp [-0.00066, +0.13553], radius +0.08172 pp
[-0.01925, +0.21567]. Both lose utility against unguarded, and the
direct learned-minus-radius lower is -0.03578 pp
[-0.09404, +0.01192]. These are independently calibrated policies
at equal total switches, **not** a matched reopened-action count
comparison. Learned abstains from reopening in 22/42 cells; radius
does so in 14/42. Neither meets the full observed gate.

The known-hard diagnostics retain a gain over the ten-source base:
+0.09889 pp [+0.02294, +0.20399] for learned and +0.12653 pp
[+0.01328, +0.27595] for radius, versus +0.30876 pp for unguarded.
These use the sum of known hard ADE errors within each source and
do not include missing outcomes. The paired intervals resample 14
sources after averaging their three seeds, not individual windows.

## How Much Does Threshold Selection Leave On The Table?

After the frozen evaluation I checked every registered threshold
on the held-source outcomes **only as a diagnostic**. I did not
export or use the maximizing thresholds. An outcome-aware choice
from that grid finds a positive gain while preserving the inner
selection criteria in 23/42 learned cells and 32/42 radius cells.
Its lower vs base is +0.19060 pp for learned and +0.22432 pp for
radius. Each actual calibrator also violates at least one of its
preservation conditions on two held-source cells. The ceiling is
optimistic, uses the held labels, and is **not** the full original
risk check. It distinguishes lost threshold transport from a
complete absence of useful grid options.

This is empirical source-held threshold selection, not the
finite-sample risk guarantee in
[Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052).
The calibration sample is three exposed source groups per fold; no
independent source exchangeability or valid multiple-testing rule
for our recording ratios is established here.

**Decision.** Learning a harm score does not beat a simple radius
guard on this experiment. Both reduce easy net failures relative
to unguarded, but do not reliably recover utility and original
risk together. Improve transfer of the calibration rule before
adding a larger guard model, and do not tune on the observed outer
ceiling. No independent data, changed 2% rule, deployment, Stage5C
or SMC. This is the exposed image-coordinate 8-to-12 study.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_release_guard.py fit
.venv-pytorch/bin/python scripts/run_m3w_release_guard.py score
.venv-pytorch/bin/python scripts/run_m3w_release_guard.py replay
.venv-pytorch/bin/python scripts/audit_m3w_release_guard_grid.py
```

Design `f9698320` precedes fitting. The complete native arm64
matrix takes 110.62 s with per-source checkpoints. All role
partitions were verified, 57 scoped tests pass, and score replay
is exact. [Protocol](protocol.md) | [Score replay](replay_execution.json) |
[Posthoc grid diagnostic](grid_ceiling.json)
