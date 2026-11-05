# Where Regression Gains Fail To Change Decisions

I replayed the verified original120 and ego-frame121 forests on the same exposed
own-B source-held records. No model was trained in this audit and no new held
split was opened. Features, targets and row IDs were checked against saved
checkpoint bindings before each prediction. Three synthetic tests passed.

| Context | Known proposals | Accepted by ego-frame | All-risk-only rejects | Easy-risk-only rejects | Both-risk rejects |
|---|---:|---:|---:|---:|---:|
| critic0 | 5767 | 7 | 0 | 1993 | 3767 |
| critic1 | 3715 | 13 | 2 | 1220 | 2480 |
| critic2 | 38579 | 0 | 0 | 621 | 37958 |

The source-macro normalized MSE reductions in absolute units are0.109302,
0.0207942 and0.604508. Contributions from unchanged rejects are0.109319,
0.0207943 and0.604508: the very small switched-set changes are slightly worse
in the first two contexts. All of context2's regression improvement leaves
the decision unchanged. Thus lower MSE does not establish useful admission.

## Support And Harm

The seven accepted source048 rows have no positive known easy reference. Their
easy risk remains NA, not a pass. The13accepted source076 rows contain six
joint-unsafe labels; normalized known easy harm/reference is3.674577%, above
2%. This is descriptive known-row, query-weighted, causal-normalized evidence,
not the original raw recording-level ratio, a population guarantee or a new
full-policy evaluation. Unknown proposals are not inserted as safe labels.

Current neural direct-risk code uses a pointwise veto plus predicted R/ER
support checks. The earlier complete scene allocator is a distinct decision
operator. The information probe omits those support checks and evaluates only
known frozen proposals. None of these operators should be silently conflated.
This audit does not rerun the full allocator or recover missing reference data.

## Model Follow-Up

Nearly every rejected row has positive predicted easy excess. This motivates,
but does not prove, a shared-partition bottleneck: a single forest currently
uses the same splits for all-risk and sparse easy-risk targets. The next fixed
contrast separates the two output partitions at the same total64-tree budget,
with a32-tree shared control. Features, signed targets, weights and zero gate
remain unchanged. No threshold relaxation, new test or claim of safety follows
from this diagnosis. Genuine unpredictable tail risk remains an alternative.

## Reproduction And Scope

` .venv-pytorch/bin/python scripts/run_m3w_moving_boundary_audit.py `

PID44790 exited0 in113.498s; peak RSS7,666,286,592bytes. Frozen source packing
contains195286rows, while the fitting-only cache contains48061known proposals
and records1274unknown proposals separately. The derived cache stays private.
The local run was reasonable; no new CREATE job was submitted.

- fresh_run: boundary attribution and fitting-cache extraction.
- cached_verified: two sets of eight already trained forest predictions.
- not_run: new training, full-policy risk, nested-C/H or independent confirmation.
- Original2% all/easy harm, easy-net, support and empty-control NA remain.
- No identity certification. Original24-group primary remains NA, not passed.
- Native obs8/pred12 silver image coordinates, not metric/seconds/true3D/foundation.
- No Stage5C or SMC. Deployment and submission readiness are unchanged.
