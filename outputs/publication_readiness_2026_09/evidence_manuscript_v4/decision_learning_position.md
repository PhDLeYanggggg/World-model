# Decision-Learning Prior Art and the Remaining Method Gap

Checked7 October2026 using original papers. This is a targeted claim audit, not
a systematic review. No model was fitted or evaluated for this update.

## Original Sources Read

- Elmachtoub and Grigas: arXiv v5, introduction, Section2 formulation and the
  SPO/SPO+ definitions and properties. Their loss evaluates the decision induced
  by a predicted cost vector, not only parameter error. The stated formulation
  has a known feasible region and a linear cost objective. The author publication
  record verifies Management Science68(1):9-26,2022.
  [Paper](https://arxiv.org/pdf/1710.08005v5),
  [author publication record](https://grigas.ieor.berkeley.edu/publications/).
- Donti, Amos and Kolter: NeurIPS2017, Sections1-3, especially the task-loss and
  stochastic-program formulation. The training target is the induced task cost;
  the paper includes constrained decisions. It therefore should not be described
  as merely another generic regression loss.
  [Proceedings paper](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf).
- Wilder, Dilkina and Tambe: AAAI2019, problem description and general framework.
  The method trains a predictor for combinatorial decision quality using a
  continuous relaxation. Its motivation already distinguishes predictive accuracy
  from usefulness to the decision problem.
  [Proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/download/3982/3860).

These reviewed sections support the stated overlap. We did not reproduce their
experiments, independently verify all their proofs or conduct an exhaustive
literature search. Competitor execution remains `not_run`.

## Consequence for This Draft

The generic claim that lower MSE can coexist with worse decisions is not novel.
Nor are cost-aware routing, deferral, reference fallback or optimizing downstream
decisions. The draft now acknowledges these precedents in its main related-work
section instead of relying only on trajectory and risk-calibration references.

Our completed evidence is narrower:216 matched cost-head fits, fixed controls,
paired locality uncertainty, explicit missing-outcome support, and a TRAIN
selected-harm decomposition. These are development findings in one protocol.
The decomposition is an accounting identity with empirical measurements, not a
new causal theorem. The negative temporal result does not show that decision-
focused optimization cannot work or that our method outperforms it.

For a positive method paper, matched decision-learning controls remain missing.
Any adaptation must retain the same forecast producer, causal inputs, training
roles, sampling budget and evaluation cohorts. Learning constraint coefficients
and dealing with unknown labels require explicit treatment; borrowing an
optimizer or regret loss does not establish selected-risk calibration.

## Next Experimental Boundary

First finish the already-registered one-factor easy-harm loss comparison and its
unchanged readout. Do not add a favorable comparator or rewrite primary tests
after results arrive. Depending on that result, a subsequent separately frozen
experiment can test a task-aligned training objective against the existing
quadratic/deviance controls. It must distinguish utility optimization from a
valid risk-calibration procedure and retain undefined support as a failure.

This note changes literature positioning only. It neither opens independent
roles nor registers or executes a new training experiment. Current results do
not establish submission readiness, metric/seconds semantics, human-gold labels,
true3D or foundation-model capability. Stage5C and SMC remain off.
