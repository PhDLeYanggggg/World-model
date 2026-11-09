# Causal Composition at Equal Query Counts

Can a causal pair term turn the frozen expanded candidate capacity into
better joint choices? Compare independent row scores, a common candidate
family per query, additive predicted pair terms, and pair-coupled choice.
All use exactly the same candidate bank, fixed cap eight, and exactly the
same realized edit count on every query. This isolates choice, not safe
admission. CV remains unchanged on unselected rows.

- Train only on the original nine nonempty fitting sources. Use the bound
  PCT/EqMotion donor arrays where each producer excluded its target sources
  and all A sources. Fit labels are the native annotation grid and masks;
  outcomes never enter the feature builder. No A targets until all causal
  action arrays are frozen. The fixed A forecast bank used 4,000 updates,
  unlike 600-update donors; keep this distribution difference explicit.
- Pointwise head predicts paired lower gains to CV and OLS4 (increments
  from CV), plus both original positive-harm budget excesses. Inputs are
  past kinematics/backtests, seven causal paths, and candidate identity.
  Scores average predicted gains and penalize positive excess. Choose one
  candidate per row. A uniform arm may use a single family where it can
  exactly match the count; otherwise it uses the same independent choices.
- A separate edge head predicts full-grid lower pair-displacement gain vs
  CV, using relative histories, candidate displacements, and four-nearest
  current-visible edges. Train on label-blind sampled candidate pairs and
  the two corresponding solo edits. Additive and joint share the same head,
  bank and solo terms; joint also adds the nonadditive pair term when the
  other edge endpoint is chosen. Fixed pair weight .25, no held tuning.
- Three seeds, 1,200 optimizer updates per head, 128 training rows or edges
  per batch, source-balanced samples, 100-step checkpoints. Evaluate the
  34 exposed A recordings with the original CV-referenced 2% rule and paired
  source bootstrap after averaging seeds. Primary joint vs additive: paired
  full-grid lower agent and pair gain. Strong CV/OLS4, direct neural and
  independent controls remain. Keep signed easy and unsupported views. A
  secondary same-displacement comparison retains the chosen tracks and
  shrinks all four arms to the smallest total peak in each query, without
  any label input. Candidate paths within `1e-8` pixels of CV are ineligible
  for all arms; the original risk rule itself does not change.
- Mandatory matched edits are not deployment. No independent confirmation,
  gate change, threshold search, metric/seconds claim, Stage5C, or SMC.
