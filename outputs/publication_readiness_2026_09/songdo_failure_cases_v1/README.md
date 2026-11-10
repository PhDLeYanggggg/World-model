# Songdo: three cases behind the aggregate result

The registered evaluation used 40 calibration flights and 160 confirmation
flights across 14 confirmation source components. In the released-policy
evaluation the joint rule made no edits; the independent rule edited 15 of
480 flight-seed views and met both utility and original risk requirements on
four. This figure uses the separately reported **matched counterfactual
proposals**, not edits made by the released policy.

I took the lower median from each [eligible set](protocol.md) at seed 17.
All 160 previously scored flights were screened via score receipts and label
masks. Numeric labels were rejoined for three flights only; every plotted
endpoint agrees with the archived contrast within 1e-12 pp. No new cohort was
opened and no model or threshold was changed.

| Example | Eligible flights | Observed gain | Missing-label bounds |
|---|---:|---:|---:|
| Different recipients | 94 | -0.1761 | [-0.1761, -0.1761] |
| Same recipients, different paths | 134 | -0.0736 | [-0.0736, -0.0736] |
| Incomplete future labels | 2 | +0.0282 | [-0.0325, +0.0890] |

Units: 0.001 percentage points of whole-flight observed CV error. Positive
means the joint-rule proposal reduces error relative to the independent rule.
In the first example the recipient contribution is -0.1966 and the path
contribution +0.0205 in the same units. In the second, keeping the two edited
actors fixed still leaves a loss. The third has a positive observed gain but
unknown sign after allowing for the missing steps. The bounds describe
missing labels, not statistical confidence. The eligible categories may
overlap; they are not prevalence estimates. These cases illustrate mechanisms
and do not change the main result or the 2% release decision.

## Reproduce

Use the original Songdo local caches and frozen proposal/score receipts,
with NumPy, PyTorch and Matplotlib installed. The review packer also uses
`pypdf`. Run from the project root:

```sh
.venv-pytorch/bin/python scripts/audit_m3w_songdo_failure_cases.py
.venv-pytorch/bin/python scripts/build_m3w_songdo_failure_case_supplement.py
```

The figure is written to
`data/stage_cvpr2027_experiments/songdo_failure_cases_v1/failure_cases.pdf`.
The anonymous `review_supplement_v34.zip` in `submission_admin` accompanies
the unchanged v201 manuscript. Its verifier checks the figure hash and
anonymous arithmetic, including the inherited v33 replay, without raw labels.
See [case data](readout.json), [review receipt](review_receipt.json), and the
[confirmation receipt](../songdo_calibrated_20261110/confirmation_receipt.json).

Data: Fonod et al., [Songdo Traffic Dataset v2](https://zenodo.org/records/17924857)
(CC BY 4.0); [dataset paper](https://doi.org/10.1016/j.trc.2025.105205).
