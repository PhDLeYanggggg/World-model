# Frozen-Proposal Peer Context

## Hypothesis And Prior Negative Evidence

The current relational head averages each peer's same-index hypothetical
candidate, although the frozen proposer can choose different candidates for
different agents. A causal fitting-input check found mixed proposals in46/52,
99/137 and735/898 queries. This is a verified representation difference, not
proof of the cause of prediction failure. Queries are overlapping windows,
not independent scenes. The proposed remedy must be tested, not assumed.

Earlier selected-query, query-unit and current-policy critics already failed
their safety/robust-utility checks. Set pooling itself is not new. In particular,
changing the current supervised set worsened completion-lower utility; repairing
query units did not establish safety. This trial does not repeat those query-loss
changes: it retains the current bound-aware two-head joint objective and fixed
source-disjoint proposer, changing only one added peer-context branch. Previous
negative results stay in their original reports. Source-weighting also failed.

## Matched Intervention

Start from the same three completed2,000-update relational heads. Add a zero
initialized64x64 bias-free bridge to each risk/decision head. Both arms have
76,548 parameters, identical initial predictions and525 fields per action:
the existing524 plus one explicit causal proposal bit. Same original context
remains in both arms. Both new bridges learn, so capacity is matched.

- same_action: bridge receives other-agent mean hidden state for the same
  candidate index as the ego candidate.
- proposed_action: bridge receives other-agent mean hidden states of their
  frozen proposed candidates, excluding ego and non-proposing peers. No active
  peer means zero bridge context. This is NOT final post-veto action context.

No source ID, future outcome, label-availability or easy-label feature. The
legal-action mask is NOT the proposal mask. Training uses verified frozen
single-proposal masks. Inference explicitly recomputes the same source-disjoint
proposer from causal inputs and the full legal-action mask, before outcomes.
Audit own-B replay of this contract before fitting. No test endpoints/goals.

Both arms use original joint loss, unknown-bound supervision, own-B scalers,
AdamW lr1e-4/weight-decay1e-4, clip5, batch72, matched query draws, symmetric
optimizer reset and1,000 additional updates. Fixed last iterate, checkpoint100.
All encoders and bridges must update. No coefficient or threshold search.

## Evidence And Stopping Rule

Real largest-context pilot:50+50 resumed versus100 uninterrupted updates for
both arms. Independently repeat the first treatment's full1,000 updates. Freeze
all six primary fits before any additional exposed nested-C readout.

Internal follow-up screen requires at least2/3contexts with >=1% reduction in
known joint sign cost, nonworse false-safe cost and nonworse risk MSE, treatment
versus matched control. This is a fitting screen, NOT scientific safety. If it
fails, keep all fits and diagnose; no extra nested-C or H readout. If it passes,
use the existing fixed readout and report all controls, missing-label completion
bounds and per-source effects without selecting thresholds/checkpoints on them.

Original2% all/easy positive-harm, signed easy-net and reference/support criteria
are unchanged. Empty support is NA, not pass. No identity-fallback certificate.
Original24-group primary remains NA. SOURCE/own-B/nested-C are already exposed
development; three contexts share producer seed17, not three independent seeds.
Moving20, DUT, DroneCrowd and H stay closed. No deployment or submission claim.
Native obs8/pred12 image coordinates; not metric/seconds/true3D/foundation/gold.
Stage5C and SMC remain disabled.

## Positioning And Execution

Permutation-invariant pooling is established prior art, not this project's
novelty claim: [Deep Sets, NeurIPS2017](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
The question is whether proposal-conditioned information improves the utility
and harm estimator under a fixed intervention rule. Pooling can dilute rare
peers; frozen-proposal context can differ from final admitted context; same
input information may still be insufficient. Neither a lower fitting loss nor
an exact replay proves generalization or causal physical interactions.

Native arm64 local PyTorch, CPU4/inter-op1/workers0. Process lock, heartbeat,
checkpoint/resume; no CREATE submission required for this scale. Commit
protocol/code before fitting. Raw/cache/checkpoint artifacts remain private.
