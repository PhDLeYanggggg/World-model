# Neighbor-Slot Sensitivity

Complete own-B data; fixed original joint controls; no new V/H.

| Context | Changed admissions % | Original / averaged MSE | Original / averaged joint cost |
|---|---:|---|---|
| eqmotion_fold1_seed17_critic0 | 44.953 | 0.009564201 / 0.6163671 | 0.006459613 / 0.01274477 |
| eqmotion_fold0_seed17_critic1 | 11.971 | 0.008399096 / 1.844052 | 0.003356657 / 0.004671167 |
| eqmotion_fold0_seed17_critic2 | 3.956 | 0.6127221 / 2.22234 | 0.01963634 / 0.02180885 |

Sensitive contexts: 2/3. Mean prediction improves both fitting criteria: 0/3.
Prospective permutation-aware training screen: **False**.

Sensitivity is not proof of a bug: the original nearest-neighbor order is a valid convention.
All shifts, clipping and error counts are retained in JSON. Original 2% safety/support
requirements are unchanged. No identity certification or new deployment result.
Native obs8/pred12 image coordinates; Stage5C/SMC off; independent confirmation not_run.

[Protocol](protocol.md) | [Full metrics](analysis.json)
