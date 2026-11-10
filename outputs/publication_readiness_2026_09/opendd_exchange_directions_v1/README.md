# A better direction, at the same correction size

I pointed the incoming edit toward a short or long velocity fit,
bounded acceleration, or a decaying turn. Every step retains its
original distance from CV. This controls both the peak and cumulative
deviation without forcing the direction or the endpoint to be the same.
Only the observed eight steps are used.

| Menu, with the option not to exchange | Positive dual passes | Gain > 0.10 pp | Gain > 0.50 pp | Deletion-stable rescues |
| --- | ---: | ---: | ---: | ---: |
| Original | 181 | 68 | 4 | 93 |
| Short velocity | 184 | 67 | 4 | 95 |
| Long velocity | 184 | 70 | 4 | 98 |
| Bounded acceleration | 187 | 77 | 3 | 102 |
| Decaying turn | 189 | 63 | 3 | 106 |
| Causal suffix rule, one direction per exchange | 191 | 77 | 4 | 108 |
| All five directions | 216 | 95 | 4 | 136 |

The oracle enumerates the same 7,198 exchanges over 606 views and
780 edits. It can choose any of the permitted directions at the
receiving edit, not new recipients. This expands the 0.10-pp
frontier by 27 views, much more than the timing changes. The
0.50-pp frontier does not increase.

At the 101 exchanges the frozen model actually chooses, acceleration
gives 60/606 dual passes, turn 57, and the observed-suffix rule 55,
against 47 for the original and 53 for exact uniform choice among
the five directions. No swaps also gives 60. For the suffix rule
the paired conservative difference against the original is
-0.000342 pp (95% location interval [-0.002480, +0.001413]).
Turn improves this contrast by +0.000991 pp
[+0.000004, +0.002243], but does not beat the no-swap count.
The direction bank makes more useful paths available in hindsight;
the simple decision does not establish a release.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_directions.py
```
