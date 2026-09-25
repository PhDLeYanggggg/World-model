# Failure Analysis: Sparse Coupling And Lost Useful Additions

## 1. Pairwise Coupling Is Rare
Of13,824 repeated query/views,1,298 permit a nonzero half-count choice and73
permit a non-additive pair-product term at that count. Only31 unique current
queries support such a term. Joint and unary choices differ in six query/views,
covering three unique queries and12 changed agent/view identities. A solver
running successfully on thousands of other queries is not thousands of tests
of multi-agent coordination: many reduce algebraically to independent/unary
selection, a unique assignment or no intervention.

## 2. The Strong Comparator Removes The Apparent Advantage
Joint control has ten positive and one negative all-ADE interval over hash
priority. It has zero positive intervals over independent predicted-gain ranking
and zero over unary geometry. The hash control therefore cannot serve as the
only comparator for a joint-agent contribution. Optimizing a proximity penalty
can reduce that proxy by construction; it need not reduce trajectory error.

## 3. Fewer Additions Can Break Easy Preservation
The failure is at producer0/seed17/all/controller1, locality eu-locality-020.
Its67 known positive-easy rows contain27 additions removed by half-count joint
selection:21 removed predictions were beneficial and six were harmful.

Relative to the same CV denominator:

| Error component | Percentage points |
|---|---:|
| Benefit lost by removing useful additions | +1.873247 |
| Harm avoided by removing harmful additions | -1.013044 |
| Net error increase from thinning | +0.860202 |
| Full-add easy degradation | 1.394771% |
| Half-count joint easy degradation | 2.254973% |

The equation is independently reconstructed from frozen choices and labels in
easy_thinning_accounting.json. This is a posthoc explanation, not a new policy
or threshold fit. Independent, hash, unary and joint half-count variants all
reach the same worst easy value; the extra pair-product term is not the sole
cause. This locality has zero predicted added-proximity proxy across the listed
policies, so that proxy cannot identify the lost trajectory benefit here.

Across all36 views, half-count joint loses all-ADE versus full-add;27 confidence
intervals are negative. Count throttling is therefore not justified as the next
safety repair. Preserving old incumbent choices alone does not guarantee safety
when useful new choices are discarded and other harmful additions remain.

## 4. Numerical Failure Was Caught, Not Accepted
One hash solve, producer0/seed43/easy/controller1 at locality008, failed its
original-unit solution-validity check. Unary and joint were certified and
algebraically identical there, but the registered all-control fallback was
applied. The query is marked unmatched. No numerical tolerance was relaxed after
seeing outcomes; no failed solve is called optimal or an accuracy improvement.

## 5. Remaining Identification Limits
Only6,116 of the parent's318,969 indexed rows belong to this pre-existing query
population. The prior whole-population easy summary and this query-subset result
have different denominators. Repeated seeds/roles are dependent and each interval
resamples only four source localities. Unknown futures remain unknown. Agents
without indexed eight-step histories are not represented in the pair graph.

This rejects the particular registered proximity-based half-count repair, not
all possible joint-agent world models. It does not justify a geometry weight
sweep on these outcomes. The next study must address incremental harm/benefit
reliability and independent calibration with matched strong comparators.
No deployment or independent-confirmation claim follows from this experiment.
