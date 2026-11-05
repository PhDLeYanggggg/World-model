# Proposal Context Did Not Improve Decisions Consistently

## What Was Tested

Six actual neural continuations compare equal-capacity peer-context bridges.
The same-action bridge sees peers' same-index hypothetical candidates. The
proposed-action bridge sees their frozen proposed candidates and excludes
non-proposing peers. Both retain the old context, original joint loss, input
normalizers, query draws and1,000-update budget. Initial predictions exactly
match the trained parent. Encoders and both new bridges really update.

The observed input distinction is real: mixed proposals occur in46/52,99/137,
735/898 own-B queries. All training proposals replay exactly from the full legal
mask and source-disjoint causal proposer. This does not establish that the old
context was a bug, or that changing it should improve prediction.

## Main Negative Result

| Context | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |
|---|---:|---:|---:|---:|
| 0 | +0.0670 | +0.9195 | 0.0000 | +0.9309 |
| 1 | +1.0776 | -6.9754 | -29.4963 | -6.6181 |
| 2 | -0.3627 | -0.5588 | -0.3314 | -0.5655 |

Positive reductions are better. These are own-fitting, normalized costs, not
held trajectory improvements or percentage-point gains. The fixed internal
follow-up screen passes0/3. This is not merely a near-miss at1% in context0:
the other two contexts worsen overall cost and false-safe cost. No further
nested-C or H evaluation was run, and no deployment changed.

In context0, beneficial proposals falsely rejected fall2612to2589 while false
safe count remains17. That is a small useful fitting effect, not stable safety.
In context1, risk MSE improves but false-safe count rises33to37 and false-reject
count1814to1828. False-safe cost changes0.0000425298to0.0000550745; its29.50%
increase has a small absolute denominator and is not a new population-risk
estimate. In context2, false-safe count falls205to204 while cost rises0.33%:
counting fewer errors is not equivalent to reducing their severity.

## Source And Context Diagnosis

Posthoc own-B source costs in context1 worsen for all three sources. Source065
accounts for about76% of the weighted joint-cost increase:0.000144479 of
0.000189940. Its conditional cost increases0.00114936to0.00219894. All sources
remain included; this is descriptive concentration, not an exclusion rule.

Mixed-proposal-query cost worsens1.681% in context1. Single/empty-proposal-query
conditional cost increases0.00100381to0.00280082 and contributes0.000145533 of
the total increase. Empty-proposal queries have zero active fitting weight;
this combined slice does not treat empty sets as safe outcomes. The effect
therefore cannot be explained solely by peers choosing different candidates.

In context0 mixed-query cost improves1.122%, while single/empty-query cost
worsens0.336%. In context2 mixed-query cost improves only0.027%, while the
single/empty slice worsens3.838%. These dependent, posthoc fitting slices are
not new primary estimands, sampling CIs or independent confirmation.

## The Added Branch Is Active, Not Necessarily Helpful

Both new risk/decision bridge norms are nonzero in every model. Temporarily
removing the learned proposed-action bridge changes joint cost as follows:

| Context | Learned bridge | Bridge removed |
|---|---:|---:|
| 0 | 0.00638681 | 0.00411678 |
| 1 | 0.00291293 | 0.00372633 |
| 2 | 0.01869096 | 0.02223217 |

The branch affects decisions but not uniformly beneficially. Removing a fitted
component is NOT a retrained ablation. It changes a co-adapted model and cannot
identify a causal contribution, justify a deployment choice, or rescue the
failed matched contrast. No model was selected using this diagnostic.

## Limitations And Next Model Question

The proposed context changes two coupled aspects: which candidate represents
each peer, and whether non-proposing peers contribute to the added branch.
The latter may discard useful background motion. Retaining the original context
reduces but does not eliminate this interpretation issue. Proposals also precede
the final veto, so they are not the final executed set. Mean pooling may hide
rare influential peers. No new visual observations were added.

Before another fit, isolate candidate identity from peer participation on own-B:
compare complete peer representations that retain fallback peers' causal floor
paths with the current active-only context. Check prior experiments first and
use a matched model if warranted. Do not tune thresholds on the failed readout,
relax2%, remove failed sources or call empty support a pass. This is a concrete
representation question, not proof that background omission is the root cause.

## Provenance And Execution

fresh_run:6,000 primary,1,000 exact-refit and400 pilot neural updates; own-B input
audit, fitting metrics, source/context diagnosis. cached_verified: parent models,
feature schema, source partitions and preprocessing. not_run: additional nested-C,
reserved/H confirmation, new trajectory-dynamics/JEPA fitting and deployment.
The three contexts share one producer seed, not three independent training seeds.

Training PID33697 exits0 in680.127s, peak RSS7,415,808,000bytes. Verification
PID34694 exits0 and all six initial/final metrics and attributions agree exactly.
The first supplementary diagnosis PID34774 exits1 because report generation ran
concurrently and changed its public-file manifest. No weights or scientific
metrics changed. The reporter now takes the shared lock; a regression test
guards it. Sequential diagnosis PID34875 exits0 in37.862s, preserving all25
previous public files. This orchestration failure is retained, not hidden.

Original2% all/easy positive-harm, easy-net and reference/support requirements
remain. Earlier17-view failures and original24-group primary NA remain. No
identity-fallback certification. Moving20, DUT, DroneCrowd and H stay closed.
Native obs8/pred12 image-local detector-silver data are not metric, seconds,
true3D, foundation or human-gold evidence. Stage5C/SMC remain off. This negative
model-side study does not make the paper submission-ready.
