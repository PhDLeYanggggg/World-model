# Ranking And The Decision Boundary

I recomputed the preceding nine neural risk heads on their own fitting
partitions only. Ranking metrics exclude zero targets and unknown outcomes;
an unobserved future is not a beneficial example. No thresholds were searched,
no new model was trained here, and no new held outcome was read.

| Harm-floor context | Query-weighted AUROC | Cost-weighted AUROC | Beneficial fitting mass rejected |
|---|---:|---:|---:|
| critic0 | 0.970201 | 0.994284 | 44.8971% |
| critic1 | 0.888084 | 0.976471 | 68.0999% |
| critic2 | 0.727235 | 0.904861 | 66.1492% |

These diagnostics distinguish ranking from the fixed-zero acceptance boundary.
High cost-weighted ranking does not guarantee useful acceptance or safety.
Critic0 has1,891 beneficial proposals blocked by the risk head alone. Critic2
has16,168 blocked by both heads and6,275 by the decision head alone.
SOURCE060's combined ordinary AUROC is0.638940; SOURCE073 rejects99.5955% of
beneficial fitting mass. A single explanation does not fit all contexts.

This is not proof that existing features generalize, nor that changing the
threshold would be safe. All observations are own-fitting diagnostics with
overlapping queries and one producer seed. AUC is not a safety certificate.

## Pooling Hypothesis Checked

The existing layer combines h_i with mean(h_other). In a two-agent query this
sum is identical for both actors for a common action. The largest fitting
context contains five such queries and two observed opposite-sign proposal
pairs; the other two contain none. Thus the exact alias is real but too sparse
here to establish it as the main explanation. Dense-query self coefficients
average0.9905/0.9638/0.9382. No pooling repair is claimed or deployed.

Independent self and pooled terms are an established equivariant construction,
not a new contribution. I checked Lemma3 and the architecture section of
[Deep Sets, NeurIPS2017](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf).
Its representation result does not establish safety or improvement for M3W.

## Execution And Next Test

Fresh diagnostic PID15551 completed in34.7152s, peakRSS7.202GB; PID15609
replayed all nine summaries exactly in34.4265s. Context support audit PID15800
completed in30.2861s and PID16057 replayed it in30.2810s. Four focused ranking
tests pass. Parent checkpoints/labels/schema were cached_verified. No reserved
Moving/DUT/DroneCrowd/H outcomes, extra goals or future inference inputs.

The next fixed training test compares within-query paired ranking with a
matched pointwise control, leaving the original loss anchors and 2% rules
intact. It tests discrimination without simply raising harm penalties.
See the [prospective protocol](../moving_paired_decision_v1/protocol.md).
It is a hypothesis, not a reported gain.

Native obs8/pred12 image-coordinate development only; not metric, seconds,
true3D, foundation or human gold. Original24-group primary staysNA, empty
support is not passed, no identity certification. Stage5C/SMC disabled.
