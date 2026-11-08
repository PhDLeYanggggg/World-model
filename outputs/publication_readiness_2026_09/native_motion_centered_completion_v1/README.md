# The pair ranking can still reverse with a motion-centered completion

The [synthetic deletion test](../native_completion_occlusion_v1/README.md) found higher coverage for an anchored OLS4 center on released tails. I changed only the center for eligible terminal gaps and reran the frozen joint-versus-additive sensitivity. The original 1x/2x radii, shared directions, released labels, 1,613 queries, 9,675 interventions per arm across three seeds, and official 2% rule stay fixed. OLS4 was applicable to 37,524 of 37,961 missing requested terminal positions; 37,512 coordinates changed. Interior gaps and ineligible tails kept their original centers.

| Constructed world | Joint minus additive pair ADE gain (image px) | Paired 95% source-bootstrap interval |
| --- | ---: | ---: |
| OLS4-centered reference | +0.0330 | [+0.0082, +0.0655] |
| 1x low witness | -0.0768 | [-0.1565, -0.0173] |
| 1x high witness | +0.1151 | [+0.0338, +0.2207] |
| 2x low witness | -0.1004 | [-0.2042, -0.0260] |
| 2x high witness | +0.1292 | [+0.0381, +0.2508] |

The old 1x signs were -0.0808 and +0.1043 with fixed terminal extension. Both signs therefore remain attainable after replacing that center. The positive reference is only one constructed world; it does not identify the unknown ordering. All worlds share a unique coordinate per missing recording-agent-step across windows and seeds. Each witness preserved released coordinates, respected the registered radius, and stayed within the earlier outer bounds. The observed contribution and aggregate replay agree. Intervals are conditional on the constructed world; they are not intervals for the true missing-label effect. No labels were imputed into the data or official results.

`fresh_run` for optimization, scoring and exact replay; `cached_verified` for the frozen policies, labels and original envelopes. Exposed development only. No metric or seconds claim, deployment, independent confirmation, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_motion_centered_completion.py score
.venv-pytorch/bin/python scripts/audit_m3w_motion_centered_completion.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_motion_centered_completion.py \
  tests/test_m3w_completion_occlusion.py tests/test_m3w_pair_completion_witness.py \
  tests/test_m3w_anchored_pair_completion.py tests/test_m3w_shared_pair_completion.py
```

[Protocol](protocol.md) | [Aggregate results](analysis.json) | [Replay receipt](replay_execution.json)
