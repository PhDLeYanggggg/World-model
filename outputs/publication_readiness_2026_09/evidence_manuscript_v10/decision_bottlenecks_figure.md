# Decision Bottlenecks In Exposed Development

**Figure.** A, hindsight capacity decreases as we restrict a label-aware
solver from the retrieved paths to one fixed proposal per query, and
then to proposals admitted by the original screen. Positive capacity
requires a verified feasible witness with positive conservative gain
over OLS4. B, actual past-only policies at the original or expanded
search size. Parentheses show the number of switches; the orange bars
count total positive-harm upper bounds above 2%, not every possible
failure. The green bars count passes of all original recording checks.
Both panels use 102 views: 34 exposed recordings in 14 sources
repeated over three forecast seeds. They do not show independent
confirmation or physical safety.

The plot reads only the two verified light aggregate results. It
does not open predictions, labels or external data.

```sh
.venv-pytorch/bin/python scripts/plot_m3w_decision_bottlenecks.py
```

The output is a single-page local vector PDF at
`data/stage_cvpr2027_experiments/m3w_decision_bottlenecks.pdf`.
Its page was rendered and checked. A clean directory containing only
the script and four aggregate/receipt files reproduced the PDF
byte for byte. The source and receipt are versioned; the generated
PDF is not committed.

[Hindsight source](../native_recording_witness_retrieval_v1/retention.json) |
[Actual policy source](../native_joint_path_portfolio_v1/analysis.json)
