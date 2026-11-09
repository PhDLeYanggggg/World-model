# Recording-Feasible Supervision for Whole Queries

Use the same nine fitting sources, three disjoint three-source role
rotations, three seeds, path library, and original evaluation rule.
The analytic teacher solver runs on each fold's training sources only.
It searches for CV-anchored recording plans with positive conservative
gain over both CV and OLS4 and the full original 2% and support checks.
Each witness is rescored before it becomes supervision. No neural
predictions or calibration/held outcomes enter the teacher search.

For each active query in a verified recording, make equal-count
alternatives from causal geometric rankings. Replace just that query
while keeping the other witness edits fixed. A preferred alternative
first passes the original inequalities and dual gain, then improves
the minimum conservative gain. Compare (i) an additive scorer trained
on gain-only preference, (ii) the same additive scorer trained on
the recording-feasibility preference, and (iii) an interaction-aware
nonadditive scorer with the same feasibility preference and draws.
The third has additional nonlinear parameters; this is a composition
ablation, not parameter-count matching. No teacher action is an
inference input.

Held choices come from the same causal geometric slate library plus
the prior frozen conditional-lead/asymmetric-risk slate. All options
use the same per-query count as that prior policy. Compare every
selector at equal counts, including the frozen policy. Record original
risk passes, worst signed easy, CV/OLS4 and neighbor geometry contrasts
with 3,000 paired source draws after averaging seeds. Calibration
uses only analytic candidate paths and the same causal count policy.
Select a score cutoff by the original recording-risk and dual-utility
requirements, otherwise return CV. Freeze before held scoring.
Do not open A, external calibration or independent confirmation.
