# Can The Frozen Trajectories Meet The Original Recording Budget?

All five learned fixed-count policies failed the original recording gate. To
separate a decision/model problem from infeasibility of the imposed count, use
the same CV/EqMotion trajectories, all 34 exposed A recordings and all three
actual forecast seeds. Score a future-label oracle, never an inference model.
No new paths, model fits, easy definitions, thresholds or external outcomes.

For each recording, optimize the original inverse-query-size weighted paired
completion-lower utility versus CV. For a known row the gain is CV ADE minus
candidate ADE; for a wholly unknown row it is minus the maximum path separation.
Budget constraints use the unchanged unweighted whole-record sums: all and easy
positive-harm excess at 2%, signed easy net at 2%, and strictly positive selected
known easy reference. Unknowns contribute their path radius to both harm bounds.
Require every selected action to differ from CV, and recheck any solver output
with the exact original evaluator without a tolerance or fallback certificate.

- **Free count:** For this binary bank, the label-aware optimum takes all
  distinct known nonharmful choices. If no easy reference is supplied, add
  the individually in-budget easy row of least weighted utility cost. Any
  feasible combination must contain an easy row whose own harm ratio is at
  most 2%; taking more harmful rows cannot improve utility. This is the same
  feasibility principle as the earlier [binary feasibility study](../moving_record_feasibility_v1/analysis.json),
  now applied to these A banks and coupled to a fixed-count comparison.
- **Old fixed count:** Optimize the same objective and recording constraints
  with `min(2, n_agents-1)` per query. Use binary MILP with zero requested gap
  and 15 s per view. Retain exact status, primal feasibility, dual bound and
  any unresolved view. Do not treat a numerical postcheck failure or time
  limit as proven infeasibility. The count is a diagnostic restriction, not
  part of the original deployment rule.

Report feasible and positive views separately, all seeds, and all original
recordings. Use source bootstrap only if every required recording is feasible
and the relevant optimization is resolved; do not hide undefined views. No
oracle row actions exported. Both paths are available before future outcomes;
the oracle's choices are not. This is conditional bank headroom, not causal
learnability, physical safety, or a new independent result. Nothing deploys.
Native image-coordinate 8/12; no metric/time, Stage5C, or SMC claims.
