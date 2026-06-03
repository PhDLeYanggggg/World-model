# Stage43-CT T100 Residual Admissibility Head

- source: `fresh_stage43_ct_t100_residual_admissibility_head`
- result_source: `fresh_torch_t100_residual_admissibility_head`
- verdict: `stage43_ct_t100_residual_admissibility_positive_diagnostic`
- gate: `14 / 14`
- mode: `small`
- checkpoint committed: `False`
- deploy on current heldout t100: `False`

## Data

- train / val / test rows: `24000 / 9000 / 10000`
- augmented train rows: `168000`
- feature dim: `191`
- denied feature hits: `[]`

## Validation Policy

- selected policy: `{'alpha': 0.75, 'alpha_index': 5, 'gain_threshold': 0.8, 'harm_threshold': 0.5, 'delta_threshold': -0.01, 'force_easy_floor': False}`
- searched candidates: `1750`
- safe candidates: `1750`
- validation t100 improvement: `0.0012`
- validation easy degradation: `0.0000`

## Test Once

- protected t100 improvement: `0.0008`
- protected hard/failure improvement: `0.0008`
- protected easy degradation: `0.0000`
- protected switch rate: `0.0670`
- ungated alpha=1 t100 improvement: `-0.0213`
- ungated alpha=1 easy degradation: `0.2023`

## Interpretation

- This trains a residual-admissibility head over CS residual candidates instead of searching only raw gain/harm thresholds.
- Labels use future waypoints only for supervised training/evaluation; inference inputs are causal CS diagnostics, residual norms, latent state, and history/goal/baseline features.
- Current heldout t100 remains floor-only unless this admissibility policy clears stricter heldout gates.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
