# Proposal-Conditioned Context: Matched Negative Evidence

We compare two neural context bridges initialized to preserve a completed risk
critic's predictions exactly. The control pools same-index hypothetical peer
candidates; the treatment pools the peers' frozen proposals. Both have76,548
parameters, original bound-aware joint supervision, matched draws and1,000
additional updates. Six primary fits, a full repeat and exact resume checks
complete. Training and inference use the same source-disjoint proposer; future
labels do not enter model inputs. The legal mask is not a proposal mask.

Across three fitting contexts, treatment-minus-control risk-MSE reductions are
0.067%,1.078%,-0.363%, and joint decision-cost reductions are0.920%,-6.975%,
-0.559%. False-safe cost is unchanged in the first context but rises29.496%
and0.331% in the others. This does not support consistent decision improvement.
The prespecified internal follow-up screen fails0/3; additional nested-C and
independent confirmation remain not_run. No held trajectory gain is inferred.

Posthoc decomposition localizes most of the second context's worsening to one
source and to single/empty-proposal windows, rather than uniquely to mixed
candidate choices. Empty windows carry no active fitting weight. The new
branch is active, but removing a co-adapted branch is not a retrained ablation.
Candidate identity and removal of non-proposing peers are coupled; isolating
them is a subsequent model question, not a posthoc positive claim.

Permutation-invariant set pooling is established prior art, not the novelty
claim ([Deep Sets](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)).
Learning a predictor and a defer/reject decision is also established
([Mozannar and Sontag, ICML2020](https://proceedings.mlr.press/v119/mozannar20b.html)).
This experiment neither reproduces those papers' algorithms nor inherits their
theorems. The research question remains reliable relative utility/harm decisions
under fixed scene/recording risk requirements, not architectural novelty alone.

Results are exposed own-B fitting diagnostics with one producer seed and
dependent queries, not independent multi-seed confirmation. Original2% all/easy,
signed easy-net and support requirements remain. Original24-group primary is
NA; empty controls are not passes and no identity exemption exists. The prior
negative safety results stand. Native obs8/pred12 image coordinates do not imply
metric scale, seconds, true3D or foundation capability. No new dynamics, JEPA,
Stage5C execution, SMC or deployment claim is made.

This addendum is separate from the compiled v9 PDF. That PDF's source, hash and
render receipt are unchanged; it does not yet incorporate this experiment.
[Full results](README.md) | [Failure analysis](failure_analysis.md) |
[Protocol](protocol.md) | [Reproduction](operation_zh.md).
