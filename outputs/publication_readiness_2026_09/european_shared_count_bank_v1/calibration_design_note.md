# Why another confidence threshold is not yet a risk method

This is a mathematical counterexample and a primary-source design note, not
a newly trained/calibrated model, outcome-dependent protocol change or proof
of generalization. The original2% complete-recording requirement is unchanged.

## Selected-reference risk is not automatically monotone under abstention

Consider two selected real actions with easy references1 and1 and positive
easy harms0.03 and0. The selected ratio is0.03/2=0.015. Remove the zero-harm
action and the ratio becomes0.03/1=0.03: fewer interventions now violate the2%
criterion. The example is checked through the unchanged actual scoring code
in `tests/test_m3w_risk_monotonicity.py`. It is synthetic arithmetic, not a
real-data experiment or a novel theorem.

A learned confidence ordering can remove reference-supporting actions first.
Consequently, nested intervention sets alone do not prove that the selected
positive-harm/reference loss decreases. Empty selected reference remains
undefined, not a zero-risk end point. The same issue applies before discussing
domain shift or missing labels. A query veto must be scored on its actual
retained cohort, with retained gain and selected support reported.

## What existing risk-control work establishes

[Conformal Risk Control, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
addresses expected risk for a monotone loss family. Our confidence-pruned
selected ratio is not automatically such a family; the example above shows why.
This does not invalidate CRC, but prevents treating its guarantee as inherited
merely by naming a threshold conformal or conservative.

[Learn then Test](https://arxiv.org/abs/2110.01052)
formulates calibration as multiple hypothesis testing. This motivates checking
a fixed candidate family against a clearly specified risk functional, rather
than assuming monotonicity. It is not authorization to reuse exposed test
outcomes or to claim calibration from the current fitting bootstrap.

[Conformal Risk Training, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6559542f75b4452ebaaf82094c7defb7-Abstract-Conference.html)
studies differentiable risk-calibrated training and tail-sensitive OCE risks.
Training for calibrated performance is therefore an existing topic, not a
novel contribution simply from adding a risk head. Applying such ideas here
would require an explicit valid unit, loss, dependence treatment and calibration
design; no such guarantee is established in this experiment.

## Consequences for the next implementation

1. A selected-query critic should predict the actual cost of the chosen action
   set, not just replay a pooled all-agent probability. Compare it with an
   equally trained row-local control on the same fitting labels. Query-level
   features must not contain eventual label availability or future endpoints.
2. Query retention needs complete-policy assessment. Predicted2% moments,
   empirical fitting MSE and a favorable average are not calibrated certificates.
   Unknown selected rows and undefined easy denominators cannot be removed
   from scoring to make the model look safe.
3. Any later statistical calibration must first specify the target: a population
   expectation/violation probability is not the existing requirement that every
   required observed recording pass. Do not silently substitute one for the
   other, relax2%, or call repeated overlapping windows independent samples.
4. The12 source localities are exposed. Reserved external roles need admission
   and an exposure-aware lineage check; historical DUT predictions preclude
   calling those records pristine. No reserved outcomes are opened by this note.

The next critic is not yet registered or run. These checks narrow its design;
they do not constitute a calibration result or an excuse to stop experiments.
No safety/deployment promotion, Stage5C execution, SMC or unit upgrade.
