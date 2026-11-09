# Unscored Songdo forecasts and exact matched-search check

I generated candidates from the [Songdo observed-history pack](../songdo_prefix_pack_v1/README.md)
before choosing an external evaluation design. The input bank is common
to any later zero-shot or calibration protocol. This run did not read
the future index or any future coordinate and did not choose a policy.

The bank contains CV, OLS4, median four-step slope and a direct neural
self-history forecast. The neural path averages the same seven OpenDD
location-excluded models within each of three seeds (17, 29, 43). I
verified all 21 checkpoint hashes and training identities and ran the
actual Torch forward pass on 81,000 Songdo observed histories. These
checkpoints were trained on pedestrians; Songdo is a vehicle dataset.
I treat this as a cross-category stress candidate, not a strong vehicle
forecaster or evidence of successful transfer.

## Two edits at the same radius

For a label-free search check, both slots take the same radius: a
quarter of the median positive per-actor minimum across the three
non-CV candidate families. This uses the existing quarter-radius
constant, but the check does not set the final intervention policy.
The radius is computed from available candidate displacements, not
errors. Paths are shortened to the radius, never expanded. The two
edited actors must differ and leave at least one actor on CV. The
common-family control uses the same candidate family for both edits.

| Seed | Windows supporting both controls | Ordered two-slot assignments | Largest window search |
| --- | ---: | ---: | ---: |
| 17 | 1,992 / 1,992 | 29,827,378 | 92,728 |
| 29 | 1,992 / 1,992 | 29,555,568 | 92,728 |
| 43 | 1,992 / 1,992 | 29,606,054 | 93,340 |

An assignment count describes availability at the checked radius,
not how many edits the frozen policy would make or how many would
pass the original 2% risk rule. It is not a score against future data.

The two equal-radius positions are interchangeable for the existing
Slate head: its unary sum and pair features are symmetric. I added a
blocked search that evaluates each unordered pair once and encodes
each action once. On the densest **107-agent** query it gives the same
additive and joint trajectories as the original ordered enumeration
for all three existing seed heads. The maximum search there is
46,670 unique pairs instead of 93,340 ordered pairs. A single local
timing check took roughly 13–17 ms per seed versus 37–42 ms with the
original enumeration. This is not an end-to-end benchmark. The
same-family control is supported and no selected Songdo path is saved.
Unequal radii still need the existing general solver.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/build_m3w_songdo_candidate_bank.py
.venv-pytorch/bin/python scripts/check_m3w_songdo_equal_radius_search.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_candidate_bank.py tests/test_m3w_songdo_prefix_pack.py tests/test_m3w_slate_shared_menu.py
```

The first command uses native arm64 CPU, four Torch threads and no
worker processes. It writes per-session hash-bound checkpoints and a
PID heartbeat; repeating it reuses verified banks. `fresh_run` is
candidate inference and the search equivalence check. `cached_verified`
is the prefix input and donor model weights. The forecast arrays remain
local under `data/stage_cvpr2027_experiments/songdo_candidate_bank_v1/`.
No scoring, deployment, new threshold, external calibration or split
assignment. The original 2% rule is unchanged. [Bank receipt](receipt.json) |
[Equal-radius search check](equal_radius_check.json).
