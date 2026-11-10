# Does training on deletion-stable actions improve exchange decisions?

The [oracle stress test](../opendd_exchange_stability_v1/README.md)
finds 93 development views with a rescue surviving three background
deletions. I tested whether supervision on these more stable outcomes
improves a risk-factored ranker. The two models have identical inputs,
architecture, draws, seeds and update counts. The auxiliary loss is
defined only when a donor option passes on the full roster and has
three valid deletions. There are 7,542 such donor examples across the
21 fit cells, including 2,407 stable positives; counts across cells
can include the same donor recording.

| Ranking on the same 101 views | Original dual passes / 606 | Stable rescues | Rescued | Destroyed |
| --- | ---: | ---: | ---: | ---: |
| Frozen, no swaps | 60 | 0 | 0 | 0 |
| Previous aligned-context ranker | 47 | 4 | 5 | 18 |
| Equal-architecture risk control | 53 | 6 | 8 | 15 |
| Auxiliary model, original full-pass score | 47 | 4 | 5 | 18 |
| Auxiliary model, stability-weighted score | 52 | 8 | 9 | 17 |
| Exact same-view uniform expectation | 51.21 | 3.77 | 6.76 | 15.55 |

The equal-location stability-weighted minus control interval for
the original dual-pass rate is [-0.112, 0.000] pp; versus random it
is [-1.454, +0.147] pp. Against chance, the secondary stable-rescue
interval is [-0.012, +0.480] pp. A higher raw rescue count does not
establish a robust benefit. Ninety-three of 101 matched swaps are
at one location; all differences versus the equal-architecture control
are at that location. The [all-eligible comparison](../opendd_exchange_stability_forced_v1/README.md)
checks whether removing that inherited mask changes the conclusion.

Native arm64 training completed 42 heads, 16,800 updates in 59.3
seconds, and replay reproduced the same result in 14.8 seconds with
no updates. The original rule, all edit and displacement budgets,
and development roles stayed fixed. No external labels were opened.

[Protocol](protocol.md) | [Readout](readout.json) | [Checkpoint receipts](head_freeze.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_stability_head.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_stability_head.py --replay
```

The first command resumes from checkpoints saved every 100 steps.
Both use four Torch threads, no data workers, and a local PID heartbeat.
