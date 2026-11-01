# Why This Is A Control,Not A Standalone Novelty Claim

## Primary-Source Follow-Up

Mao,Mohri and Zhong's [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
(ICML2024) includes a fixed-predictor setting in Section4. Equation3 defines a
cost-weighted classification surrogate rather than training on an oracle's
one-hot expert identity. Its consistency statements require specified loss,
cost and hypothesis-class conditions. Section2 assumes bounded regression loss
and bounded expert costs. This makes a compatible empirical deferral adaptation
a relevant strong baseline,not evidence that generic trajectory expert selection
is new. The present moment projection does not implement that surrogate or inherit
its theorem. Nor do those statements alone establish our per-recording harm
ratios under overlapping,partly observed SOURCE outcomes.

Access update:fresh official publisher metadata and the publisher-linked
[PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/mao24d/mao24d.pdf)
were checked on1November2026. PDF pages1,3-7,including Eq3 and Theorem4.1statements,
were read locally after the web parser rejected its MIME type. This is not a full
appendix-proof audit. Private PDF SHA256:
`06d3b0b82937a157b23084ba8ba6c7110420edec0ecd7e545f57fee18a9a1ed5`.
The third-party PDF is not committed. This extends the preceding abstract-only
access record,without revising its historical disclosure.

## Our Moment Convention

For a realized nonnegative fallback error R and candidate error E_k,define
B_k=max(R-E_k,0)and H_k=max(E_k-R,0). Therefore E_k=R-B_k+H_k>=0 and B_k<=R.
The fallback error is the same whichever forecast is selected. Its conditional
expectation must also be candidate-independent when conditioning on the same
past and deterministic forecast bank. An easy event fixed independently of the
candidate gives a common R_easy and H_easy<=H. These are basic identities for
this forecast-replacement task,not a new theorem or physical-safety result.

The current postprocessor first shares predicted reference moments through an
available-candidate arithmetic mean,then caps B at R. It is a sequential repair,
not the closest joint Euclidean projection of every cost moment. H/H_easy do not
change. Reference MSE can improve from averaging alone,even if choices get worse.
No trajectory,latent world state or additional neural representation is learned.

## What Still Needs Evidence

The prospective method question is whether causal relative gain/harm learning
and query-level allocation improve complete policies at a fixed risk contract,
including missing outcomes and easy-case tails. It requires matched deferral
baselines and contribution controls,not just an architecture name or algebraic
coherence. The registered full-roster effect and actual recording risk cannot be
replaced by a favorable MSE,oracle witness or post-selected subset.

A future published-surrogate baseline must state its handling of vector-valued
trajectories,bounded costs,inference cost,missing labels,query coupling and
normalization. Silently clipping large errors would change the task. No such
baseline is reported as implemented or evaluated in this control experiment.
Independent role-compliant evaluation,forecast-seed replication and reliable
outcome support remain prerequisites for a submission claim.
