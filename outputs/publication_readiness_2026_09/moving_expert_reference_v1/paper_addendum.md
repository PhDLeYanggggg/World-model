# Addendum: Shared Causal Expert Context For Reference Learning

This addendum is separate from the unchanged compiled v9 paper. It does not
establish submission readiness, deployment safety or independent confirmation.

## Controlled Method Question

Does making the frozen expert's predicted rollout context available to an
action-invariant reference branch reduce selected reference optimism? The two
matched3,938parameter networks inherit the same reference state and either
mask or admit70 shared causal descriptors. The rest of the cost model remains
frozen. All48fits and48full exact refits precede the18context decision freeze.
Adding a context projection is not a novelty claim in isolation.

## Registered And Negative Results

The registered context-minus-history joint-free paired completion-lower contrast
is+0.133479pp, exploratory source interval[-0.140981,+0.556238]. Known-label
contrast is+0.175721pp[-0.090314,+0.594128]. Only3/8source lower means are positive,
and one of3risk-seed lower means is negative. One fixed forecast teacher seed
is shared. Bootstrap uses3,000source-component resamples, not overlapping windows.

| Joint-free reference | Known ADE gain over CV % | Supported original recording views | Known easy harm violations |
|---|---:|---:|---:|
| Frozen prior | 0.667496 | 23/72 | 27 |
| History refit | 0.614917 | 25/72 | 23 |
| Expert context | 0.790637 | 24/72 | 26 |

The higher context point does not establish a stable gain or safe candidate.
Seventy-two views repeat24recordings over3risk seeds. All required views remain,
including six undefined context easy ratios. Original2%positive harm,signed
easy-net and support rules are unchanged. No identity-fallback certification.

## Mechanistic Evidence

Context improves total-reference MSE from3.360713 to3.154554 versus the matched
refit, while easy-reference MSE worsens0.230431 to0.280262. All final matched
training minibatches improve, distinguishing fitting from transfer. Context
joint choice overpredicts selected ER in62/72views. This motivates easy-support
and conditional-reference/harm supervision as the next model contrast, not a larger
generic network or a looser risk limit. The attribution probe is descriptive,
not a claim that conditional probabilities must vanish on realized negatives.

All candidates preserve H/EH and label support exactly, but the necessary
benefit cap changes4,503repeated context candidate/seed views; this is not a
pure denominator-only effect. Same-count utility is retained in the diagnostics
and never used as safety; it can change amplitude and membership. Solver
postcheck failures retain feasible parents, not claimed optimal solutions.

## Scope

Data are already-exposed MovingDroneCrowd++ own-B, native obs8/pred12 available-
label image-coordinate ADE. They are not independent test, official complete12
ADE, SDD t50, meters, seconds, true3D, foundation or human-gold evidence. Upstream
annotation-generation causality is unverified. Original24-source-group main
remains NA. This is reference-cost learning, not a new trajectory dynamics model.
Stage5C/SMC remain off. See method_positioning.md for the retained prior negatives,
failure_analysis.md for source/seed breakdown and verification_report.md for
actual execution, numerical fallbacks and replay limitations.
