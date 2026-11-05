# Verification Receipt

Scientific analysis SHA256:
`ba6d32dbaf6f14b437e798a1cfe357ce0214575dc35896136f2f1979259725cb`.

The first process PID23437 exited1 at the identity-control metric check,
before producing an analysis. The corrected implementation uses the existing
joint metric; initial registration and repair details are retained.

Corrected run PID23566 completed in50.761928541s, peak RSS6400540672bytes.
Verification PID23707 recomputed all three contexts exactly and exited0.
Completed replay PID23912 reproduced all results in51.272077458s, peak
6380175360bytes, leaving all ten then-existing public file contents unchanged.
The latter run did not replace original execution receipts. Exit0 observed.

Three diagnostic tests pass, including full history/mask permutation inversion,
identity agreement and the original both-channel/zero/unknown cost semantics.
They are included in the combined26-case suite, not extra cases. One additional
replay-receipt guard case passes:27 distinct focused cases in total. No full
historical-suite result is claimed.

All model weights and preprocessing remain unchanged. The prospective
permutation-training screen fails; no new model, V/H evaluation or deployment
is inferred from these engineering checks. Original2% and support rules remain.
