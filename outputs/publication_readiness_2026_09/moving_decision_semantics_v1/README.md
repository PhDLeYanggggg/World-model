# Decision-Semantics Audit

I checked the fitting data before making another model change. This corrects
an explanation in the preceding composed-risk report; it does not change its
scores, failed safety result or any evaluation rule.

| Own-B context | Rows | Originally legal actions | Actually supervised proposals | Router fallback rows |
|---|---:|---:|---:|---:|
| critic0 | 6,197 | 37,182 | 5,956 | 241 |
| critic1 | 5,521 | 33,126 | 3,882 | 1,639 |
| critic2 | 50,748 | 304,488 | 39,497 | 11,251 |

Every supported row has exactly one supervised action. Each mask is freshly
reconstructed from the source-excluded frozen router and exactly matches the
training packet. Row IDs and causal features also match. The earlier claim
that these models average all legal actions was wrong. Proposal conditioning
was already implemented and previously tested; it is not a new repair.

Among observed proposed actions, opposing all/easy target signs occur zero
times in all three contexts. That proposed explanation is also unsupported.

| Context | Target-safe observed proposals | Rejected by composed model | Joint false-reject cost | Joint false-safe cost |
|---|---:|---:|---:|---:|
| critic0 | 5,303 | 2,987 | 0.00742947 | 0.00007322 |
| critic1 | 2,248 | 2,073 | 0.00403215 | 0.00004070 |
| critic2 | 24,216 | 23,981 | 0.02805911 | 0.00003831 |

Joint acceptance requires both heads and both channels to be nonpositive.
For this descriptive diagnostic only, a jointly unsafe label costs the sum
of positive normalized channel excesses; a safe label costs the sum of their
negative magnitudes. This is not a revised scientific metric, a probability,
a population risk certificate, or evidence that unseen proposals are safe.
The model still rejects many target-safe proposals on its own fitting data.
The next model diagnosis is which output blocks these rows, including exactly
zero target channels that receive no signed-cost gradient.

## Evidence Scope

`analysis.json` is fresh fitting-only recomputation; parent data, schemas,
routers and checkpoints are hash-verified cached assets. No new training,
validation or H readout. An initial module-reference error is retained in
`attempt1_failure.md`; `registration_v2.json` binds the corrected script.
Exact replay is recorded separately in `verification.json` when completed.
Three focused synthetic tests pass; they are not real-data safety evidence.

Original 2% all/easy risk, easy-net and support requirements stay unchanged.
No identity certification; empty controls stay NA and old failures stand.
Original24-group primary remains not estimable, not passed. Reserved Moving,
DUT, DroneCrowd and H remain closed. Native obs8/pred12 image coordinates only;
not metric, seconds, true3D, foundation or human-gold evidence. Stage5C/SMC off.
No deployment or submission-ready claim.

Run with native arm64 `.venv-pytorch/bin/python`:

```sh
.venv-pytorch/bin/python scripts/diagnose_m3w_decision_semantics.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_decision_semantics.py
```
