# Can the rare reference-loss event be learned in two parts?

I refit the model from the [single-query experiment](../opendd_counterfactual_support_v1/README.md)
with a local-loss head and a conditional last-reference head. Their
probability product estimates whole-record loss. The conditional term
trains on local losses only. The same model also runs with its local
head alone, isolating the effect of using the product at inference.
The 21 fits use 12,600 updates; histories, candidate menus, query
budget and the original evaluation stay unchanged. This is a
post-result development follow-up, not independent confirmation.

| Policy | Queries replaced | Lost / restored reference | Risk passes | Dual passes |
| --- | ---: | ---: | ---: | ---: |
| Direct whole | 365 | 27 / 13 | 112 | 84 |
| Factorized whole | 360 | 16 / 11 | 117 | 86 |
| Same fitted model, local head only | 347 | 13 / 11 | 121 | 91 |
| Gain, matched to factorized queries | 360 | 24 / 15 | 116 | 88 |
| Past prefix | Not matched | 0 / 0 | 124 | 90 |

Factorization reduces lost references relative to direct whole, but
does not make better decisions than the local head of the same model.
It loses **0.057 conservative pp [-0.164, -0.00034]** versus that local
readout. At identical replacement times it prevents eight losses versus
gain alone, but loses two dual passes. All arms keep the same 1,457
original edit counts and radii. Worst signed easy degradation remains
15.13% in both factorized readouts. The fixed probability cutoff is
not [conformal risk control](https://arxiv.org/abs/2208.02814).

Next I would use a matched training target for actual dual-pass gains
and losses, rather than reference survival alone, and first check its
support on additional permitted development sources. OpenDD's usable
replacement opportunities are too concentrated to justify repeated
threshold searches. No deployment change or independent confirmation.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_factored_counterfactual_support.py --replay
```

Omit `--replay` to fit or resume. [Protocol](protocol.md) |
[Frozen metrics](readout.json) | [Manuscript v153](../evidence_manuscript_v153/README.md)
