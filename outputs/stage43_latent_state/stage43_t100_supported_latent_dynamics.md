# Stage43-CR T100 Supported Latent Dynamics

- source: `fresh_stage43_cr_t100_supported_latent_dynamics`
- result_source: `fresh_torch_t100_supported_latent_dynamics`
- verdict: `stage43_cr_t100_supported_latent_dynamics_keep_floor`
- gate: `14 / 14`
- mode: `small`
- checkpoint committed: `False`
- deploy on current heldout t100: `False`

## Data

- train / val / test rows: `24000 / 9000 / 10000`
- feature dim: `162`
- feature hash: `a8853644a3ee29d32f031e8fef259bf6654543e50a7f6921fb174d635121a5d0`
- denied feature hits: `[]`

## Validation Policy

- selected policy: `{'gain_threshold': 0.0, 'harm_threshold': 0.1, 'failure_threshold': 0.0}`
- validation t100 improvement: `0.0000`
- validation easy degradation: `0.0000`

## Test Once on Supported Protocol

- protected t100 improvement: `0.0000`
- protected hard/failure improvement: `0.0000`
- protected easy degradation: `0.0000`
- protected switch rate: `0.0000`
- ungated t100 improvement: `-0.1938`
- ungated easy degradation: `0.7697`
- latent variance: `0.184954`

## Interpretation

- This is a t100 supported-protocol neural diagnostic, not a current heldout deployment.
- Current heldout t100 remains floor-only until a model passes source/scene support and heldout safety gates.
- Future endpoints/full waypoints are labels only; inputs are causal history, goal prototypes, baseline rollouts, floor rollout, domain/horizon tokens, and current state.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
