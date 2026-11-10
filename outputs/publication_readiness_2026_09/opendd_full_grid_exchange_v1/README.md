# OpenDD: full-grid risk leaves most exchange headroom intact

I replayed every one of 7,198 legal exchanges plus the incumbent on
202 previously scored OpenDD development recordings at seven locations,
with three training seeds. The 780 edits, forecast bank, per-recording
movement caps, and the aligned ranker's 101-view switch mask stayed fixed.
The stricter column requires the original dual pass and the full-grid
risk certificate. Original conservative utility already bounds the gain
against both motion references on all twelve steps.

| Choice | Original dual pass / 606 | Also passes full-grid risk | Also gains >0.10 pp against both references |
| --- | ---: | ---: | ---: |
| Frozen incumbent | 60 | 55 | 20 |
| Aligned learned ranker | 47 | 44 | 17 |
| Uniform exchange on the same mask, expectation | 51.21 | 46.73 | 14.80 |
| Hindsight best on the same mask | 102 | 98 | 43 |
| Hindsight keep or best exchange on any view | 181 | 175 | 63 |

Of 1,106 individual options that originally pass, 907 survive. Among the
199 losses, 193 can exceed the easy limit, 140 can lack a selected-easy
reference, and 52 can exceed the all-edited limit. These reasons can overlap.
Most affected views have other passing options: the best available plan
loses only six passes. There is still a gap on the original mask (98 versus
44), and a larger gap when hindsight can also decide where to switch.

After giving each location equal weight and averaging seeds, the best legal
choice or keep improves the full-grid pass rate over the incumbent by
**12.00 pp** (95% location bootstrap **[4.97, 18.84]**). The aligned ranker
trails the same-mask uniform expectation by **0.58 pp**
(**[-1.50, -0.03]** for aligned minus uniform). The 175 view-level oracle
passes are concentrated: 162 are at two locations. The interval measures
variation across the seven exposed locations, not generalization to a new
cohort. The oracle uses outcomes and is not an available policy.

**Implication:** On this OpenDD menu, missing positions are not the main
reason the learned ranker leaves headroom unused. The same-mask result
separates ranking from admission, without adding edits or increasing
movement. This is not evidence to deploy the ranker or lower the risk bar.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_full_grid_exchange.py
```

`fresh_run`: full-grid scoring of the original exposed development roster;
21 location-seed cells in about 11 seconds locally, using four Torch
threads, zero workers, and per-cell checkpoints and heartbeat.
`cached_verified`: original forecasts, ranker checkpoints, counts and
movement. `not_run`: fitting, CREATE, or any new independent confirmation.
[Protocol](protocol.md) | [Registration](registration.json) | [Readout](readout.json).
