# Fixed-Count Support-Aware Proposal Retrieval

Keep the original native observation-8 / prediction-12 forecast bank,
34 exposed development recordings, 14 source groups, three forecaster
seeds, and training-derived easy/hard cuts. The original 2% recording
rule and undefined-reference handling do not change. CV remains the risk
reference and OLS4 the accuracy comparator. No independent data are opened.

Fit a source-balanced ridge probability model for positive observed easy
reference, excluding the scored source from feature standardization and
fitting. Inference uses past speed and acceleration, current neighbor
spacing and velocity contrast, crowd size, and disagreement of causal
rollouts. The future-defined label is used only on fitting sources or by
the separate scorer.

At each current query, hold the stronger normalized-gain action of the
original joint proposal fixed. Compare replacements for its weaker slot:
one best estimated-gain action per eligible agent, at most eight slots,
including the original action. Two retrieval variants use this identical
shortlist, choosing by normalized gain or predicted probability of
easy reference with gain as tie break. Submit one set per query and retain
exactly the original number of agent slots. Keep the original proposal as
another control. Apply the same previously frozen gate, with unchanged
inputs and rules, to each resulting set. No held label chooses a setting.

Primary endpoint: difference in the fraction of recording views with
positive observed easy reference anywhere in the proposal roster
(support ranking versus gain ranking). Confirm usefulness separately with
a positive-lower-gain single-query witness, matched-count forecast utility
and pair geometry, and original recording-risk passes after the fixed gate.
Report the number of emitted interventions; equal proposal count does not
imply equal deployed count. Source-bootstrap intervals use 3,000 draws
after averaging seeds. Hindsight feasibility is not learned performance.
