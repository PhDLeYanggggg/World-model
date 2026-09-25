# Support, Rejection and Transport

## Material Passport

- Scope: focused primary-source reading, not a systematic review or exhaustive novelty search.
- Read on 2026-09-25; source records and relevant sections inspected. This is not a claim that the project owner has read these papers.
- Evidence role: method positioning and assumptions, not evidence that M3W improves.
- Current experiment: development-only fixed-support factorization, with no new training or independent confirmation.

## Selective Regression Is Not Automatically Subgroup-Safe

Shah et al. study selective regression and show that reducing coverage can worsen a subgroup even when the overall risk improves. Their monotonic-risk criteria and sufficient representation conditions concern conditional selective risk, with specific assumptions about conditional means and variances. I read Sections 2.2, 3.1-3.2 and 4.1-4.2. These results do not certify percentile support boxes or ADE-based neural-versus-fallback switching. [Primary paper, ICML 2022](https://proceedings.mlr.press/v162/shah22a/shah22a.pdf).

Our question differs in two ways. Rejection produces another forecast rather than no prediction, so the full indexed population still incurs loss. A rejection can therefore remove a useful neural forecast even if it lowers accepted-neural risk. Second, the locality strata in this experiment are source domains, not demographic groups; this is not an empirical fairness claim. We retain both full-policy error and source-level degradation. Same-frame count-matched controls test allocation at equal coverage, not the paper's sufficient representation conditions.

## Removing A Feature Does Not Establish Transportability

Subbaswamy, Schulam and Saria express anticipated changing mechanisms in a causal selection diagram and derive invariant interventional predictors where identifiable. I read the introduction and the graph-pruning/graph-surgery development through Section 3. Identification depends on causal assumptions; removing an unstable-looking feature alone is not equivalent to graph surgery. [Primary paper, AISTATS 2019](https://proceedings.mlr.press/v89/subbaswamy19a/subbaswamy19a.pdf).

The present intervention is on an algorithmic decision rule: hold boxes, sources and forecasts fixed, then project different box axes. It measures which rule components remove benefit. It does not change producer training size or identify a causal data-generating graph. Even if disagreement-only rejection dominates, that cannot establish that the two-source fitting versus four-source evaluation producer difference caused the failure. Producer-matched retraining would be a separate experiment, with exclusion and training-budget controls.

## Scope Of A Defensible Contribution

The useful contribution to develop is learning and allocating improvement relative to a real fallback, not the generic act of abstaining or combining JEPA and a Transformer. The present accounting is a falsification instrument: every support component must justify the good forecasts it discards as well as the errors it prevents. It is not yet a novel transport theorem, independent generalization result, calibrated risk bound or proof of joint-agent reasoning.

All trajectories here remain image-pixel obs8/pred12 with raw annotation stride 12. Detector-derived labels are not human gold. Stage5C/SMC are off; no metric, seconds, physical-safety, true-3D or foundation claim is made.
