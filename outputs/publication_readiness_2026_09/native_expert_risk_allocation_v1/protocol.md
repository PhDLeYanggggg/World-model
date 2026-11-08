# Held-source expert utility and joint allocation

The previous forecast mixture did not beat anchored OLS4 on the full
grid, but the five fixed experts and their ten pair averages contain
substantial hindsight headroom. Keep that menu fixed. OLS4 is the
identity action; only actual changes are eligible. Fit per-candidate
signed full-grid gain, non-cancelling harm, and easy harm from causal
history, in-history replay, neighbors and forecast-change size.
Easy training membership uses the positive CV-error lower quartile
computed inside each donor fold. Targets are relative to OLS4 and
include per-step disagreement for unavailable labels. Outcome values
and masks are never model inputs.

Train matched gain-only and gain/harm networks, 500 updates each on
eight of nine fitting sources, three fixed seeds. Both use the same
initialization and sampled rows. At inference compare gain-only,
gain plus risk training using gain alone, and risk-adjusted gain
(subtract half predicted harm and half predicted easy harm). Use
exactly up to two legal changes in each query for all active controls.
Compare fixed minimum-change and recent-backtest controls. The oracle
selects from the same budget and menu with labels, for diagnosis only.

From the same risk-adjusted scores and per-agent nominations, test
independent top-two, additive contraction penalty, and joint
contraction penalty. Enumerate all pairs in the same top-12 shortlist;
there is no claim of optimality beyond it. Use predicted reduction of
pair distance relative to unchanged OLS4, not a physical collision
label. The additive rule estimates each agent's contraction with
unmodified peers. The joint rule replaces the selected pair's two
unary contributions with its actual joint contraction. The penalty
weight is 0.25, chosen before scoring. Use only current positions
and causal rollouts for this calculation.

Primary contrasts: risk-adjusted independent versus OLS4 and
minimum change; joint versus additive. Report available-label
ADE/FDE, direct full-grid paired bounds, source intervals (3,000
resamples after seed averaging), fixed-budget regret, easy harm,
unsupported sources and pair distance contraction. A forced-count
comparison does not certify the original recording-risk rule or
replace an actual full deployment test. No threshold or model
selection on these held fitting outcomes.

Keep the existing fitting roster, 8/12 native annotation steps,
image pixels and the original 2% evaluation rule. No independent
development, DroneCrowd calibration or confirmation outcomes.
No deployment, Stage5C or SMC.
