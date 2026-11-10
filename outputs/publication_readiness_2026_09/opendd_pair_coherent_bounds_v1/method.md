# A shared-position outer bound for relative-motion error

Consider two fixed proposals, with reference displacements (r_i)
and prediction displacements (p_i) at one horizon step. The observed
labels (y_j) stay fixed. For a missing label (y_i) and a known
neighbor (j), the gain in relative-motion error is

```text
||y_i - (y_j + r_i - r_j)|| - ||y_i - (y_j + p_i - p_j)|| .
```

The same unknown (y_i) occurs in all these edges. Put
(R_j=y_j+r_i-r_j) and (P_j=y_j+p_i-p_j). The sum of distances from
(y_i) to the (P_j)'s is unchanged if the anchors are permuted.
For any bijection (pi), the reverse triangle inequality gives

```text
| sum_j ||y_i - R_j|| - sum_j ||y_i - P_j|| |
    <= sum_j ||R_j - P_{pi(j)}|| .
```

Minimize the right-hand side with a linear assignment. This is the
unnormalized finite equal-mass Wasserstein-1 cost; see Gabriel Peyre and
Marco Cuturi, *Computational Optimal Transport*, 2019,
doi:10.1561/2200000073.
Matching the original neighbor identities is a feasible assignment, so
the result cannot exceed the original sum of edge debits. The derivation
uses equally weighted edges; arbitrary edge weights require a different
transport problem.

Add these bounds for all missing-to-observed stars at each time. Keep
the original triangle debit for pairs whose two endpoints are missing.
The observed contribution and the full-grid denominator do not change.
This gives a tighter outer interval valid for any completion of the
missing coordinates, without modeling the process that made them
missing. It is not necessarily sharp: two-missing edges can couple
stars, and the chosen assignment need not be achieved by any actual
position.

If all anchors of one star translate by the same vector, the identity
cost is already optimal: the norm of the sum of the displacements is a
matching-independent lower bound on the assignment cost. With
unrestricted missing coordinates the two ends of this single-star
bound can be approached far along that vector and its opposite.
The assignment cannot help when there is only one changed anchor, or
when every affected edge is two-sided missing.

To distinguish a loose interval from a real ambiguity for a single
query, evaluate explicit completions. Keep the observed labels exact.
For missing actors, try the reference and prediction paths, and two
finite shifts based on the summed direction of the observed anchors.
At each step choose the best and worst of these complete assignments.
The resulting values are feasible inner witnesses under the evaluation's
unrestricted missing-coordinate model. They are not claims about
physical plausibility. Different views may reuse labels, so do not sum
witnesses over seeds or use them to assert an aggregate attainable range.
