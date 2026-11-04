# Positioning The Critic Experiment

Targeted primary-source reading on 4 November 2026, while the fixed training
run continued. This note changes no experiment, threshold or evaluation rule.
Sections below were checked directly; this is not a claim of a complete
systematic literature review or a new risk-control theorem.

## Established Results

- **Risk-Controlling Prediction Sets (RCPS).** Section 2 separates training
  from an IID calibration sample; Theorem 1 gives upper-confidence-bound
  calibration under its stated nesting, loss and continuity conditions.
  Generic holdout calibration and user-specified risk control are established
  ideas, not M3W contributions.
  [Bates et al., arXiv:2101.02703v3, Sections 2.1-2.2](https://arxiv.org/html/2101.02703v3).

- **Learn then Test (LTT).** Section 1.1 specifies IID calibration data;
  Section 2 frames candidate-policy risk control as multiple testing.
  Section 2.4 already addresses multiple risks and multidimensional tuning
  parameters. Combining all-agent and easy-agent risk constraints does not
  alone establish a new general calibration method.
  [Angelopoulos et al., arXiv:2110.01052v5, Sections 1.1 and 2](https://arxiv.org/html/2110.01052v5).

- **Conformal Risk Control (CRC).** Sections 1.1 and 2 treat exchangeable,
  bounded-above, non-increasing loss functions and expected test loss. Its
  extensions include multiple risks. This is not automatically the same
  object as this project's per-record selected-reference ratio and support
  contract. One cannot obtain that contract merely by naming a gate conformal.
  [Angelopoulos et al., arXiv:2208.02814v4, Sections 1.1, 2 and 4.3](https://arxiv.org/html/2208.02814v4).

- **Non-monotonic CRC.** This 2026 preprint gives stability-dependent guarantees
  for non-monotonic and multidimensional procedures under exchangeability.
  Section 2.2.3 explicitly treats selective classification. Non-monotonic
  selection is therefore not, by itself, an unexplored problem. The required
  stability conditions would need a separate argument for our algorithm.
  [Angelopoulos, arXiv:2602.20151v1, Theorem 1 and Section 2.2.3](https://arxiv.org/html/2602.20151v1).

## Implications For This Project

The following are **project-specific inferences**, not theorems borrowed from
these papers:

1. Separating proposer fitting from risk-estimator fitting is a controlled
   model/data-isolation intervention. It is not independent risk calibration.
   The new heads still share a forecaster family and exposed dataset sources.
2. Repeated windows are not new independent scenes. The current 23 source
   bootstrap units have not been verified as IID physical sites. A descriptive
   interval is not a finite-sample distribution-free safety guarantee.
3. A predicted expected harm/reference ratio is not a confidence bound.
   Estimating its components accurately on average does not establish every
   selected recording's risk or positive easy-reference support.
4. Original zero-reference views remain unavailable under the user's retained
   contract. Neither a neural head nor an external theorem supplies missing
   denominator evidence. No epsilon denominator, removed view, identity
   certification or replacement primary is introduced.
5. A plausible empirical contribution still needs a useful multi-agent
   decision mechanism, matched intervention controls, improvements that
   survive uncertainty accounting, and genuinely separate confirmation.
   A JEPA/Transformer combination, seed ensemble, or source split alone is
   insufficient. This study tests one prerequisite, not submission readiness.

## What The New Experiment Can Establish

All 108 new cost heads must finish before nested-C scoring. Same-data seed
controls and source-disjoint critics use the same architecture, objective,
action bank and decision rule. Their controls match each action count in
each query. A positive matched contrast could support further study of the
estimator mechanism; a negative or unavailable contrast must remain visible.
The original full-policy risk contract is evaluated separately and unchanged.
No new held-out H result, independent confirmation, physical-safety claim,
metric/seconds claim, Stage5C execution or SMC follows from this note.
