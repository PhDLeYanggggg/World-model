# OpenDD: learning the whole list does not recover the exchange gap

I tested whether an exchange ranker can learn to choose the better of
several legal plans for the same recording. The design adapts the
[ListNet top-one cross-entropy idea](https://doi.org/10.1145/1273496.1273513):
normalize the scores within each slate, but assign target mass only to
plans that pass risk and beat both references. This is not a reproduction
of the original retrieval benchmark. All three controls use the same
features, legal alternatives, network, 32-slate batches and optimizer
budget. Each fit excludes its evaluation location. Donor slate producers
and cutoffs exclude both their donor and the outer evaluation location.

| Choice at the same 101 switch views | Original dual passes / 606 | Also full-grid | Also >0.10 pp against both references |
| --- | ---: | ---: | ---: |
| Keep the incumbent, no extra exchanges | 60 | 55 | 20 |
| Archived aligned ranker | 47 | 44 | 17 |
| Uniform choice, exact expectation | 51.21 | 46.73 | 14.80 |
| Pointwise, original labels | 57 | 53 | 19 |
| Listwise, original labels | 52 | 48 | 16 |
| Listwise, full-grid labels | 50 | 47 | 19 |

The same-mask hindsight count is 98 full-grid passes. Adding the listwise
term to pointwise training changes the equal-location pass rate by
**-0.19 pp [-0.56, 0.00]** with the same original targets. The five lost
passes come from one location. Replacing the targets with full-grid
labels gives **+0.03 pp [-0.22, +0.32]** relative to the original-label
listwise model and **-0.40 pp [-1.43, +0.24]** against same-mask uniform.
Neither preregistered comparison for the full-grid target clears zero.
The full-grid model has one fewer raw pass than original listwise but a
slightly higher location-weighted point estimate because the locations
have different numbers of recordings. Intervals use 3,000 paired
location resamples after averaging seeds; the 606 views repeat 202 recordings.

The labels actually changed: 6,094 of 7,566 original passing donor
options remain positive under full-grid risk, among 67,275 options in
4,507 slates summed across fit cells. These totals repeat donors across
folds, not independent samples. A stricter label and a within-recording
loss do not recover the unused candidate headroom with these features.
The simpler pointwise model comes closest to the incumbent, but does not
exceed it. I kept the release policy, risk threshold and v210 main text
unchanged. This is development evidence, not external confirmation.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_listwise_exchange.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_listwise_exchange.py score
```

`fresh_run`: 63 fits and 25,200 updates in 103 seconds, then scoring in
9 seconds on native arm64, four Torch threads, zero workers. The fit saves
weights, optimizer and sampler every 100 steps. Repeat `fit` to resume;
`score` uses the saved choices with zero updates. The choice and model
hashes were published at `c37b4ab6c` before scoring. A second `fit`
invocation verified all choices with zero optimizer updates. `cached_verified`:
original rosters, past-only features, action menus and all three frozen
controls. `not_run`: new external labels, CREATE, or deployment.

[Protocol](protocol.md) | [Freeze](head_freeze.json) | [Readout](readout.json).
