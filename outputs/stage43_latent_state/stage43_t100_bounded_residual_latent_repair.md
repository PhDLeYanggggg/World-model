# Stage43-CS T100 Bounded Residual Latent Repair

- source: `fresh_stage43_cs_t100_bounded_residual_latent_repair`
- result_source: `fresh_torch_t100_bounded_residual_latent_repair`
- verdict: `stage43_cs_t100_bounded_residual_latent_keep_floor`
- gate: `16 / 16`
- mode: `small`
- residual clip: `0.2`
- checkpoint committed: `False`
- deploy on current heldout t100: `False`

## Data

- train / val / test rows: `24000 / 9000 / 10000`
- feature dim: `162`
- feature hash: `a8853644a3ee29d32f031e8fef259bf6654543e50a7f6921fb174d635121a5d0`
- denied feature hits: `[]`

## Validation Policy

- selected policy: `{'alpha': 0.05, 'policy': {'gain_threshold': 0.0, 'harm_threshold': 0.03, 'failure_threshold': 0.0}, 'force_easy_floor': True}`
- searched candidates: `2520`
- safe candidates: `2520`
- validation t100 improvement: `0.0000`
- validation easy degradation: `0.0000`

## Test Once on Supported Protocol

- protected t100 improvement: `0.0000`
- protected hard/failure improvement: `0.0000`
- protected easy degradation: `0.0000`
- protected switch rate: `0.0000`
- ungated bounded t100 improvement: `-0.0225`
- ungated bounded easy degradation: `0.2129`
- latent variance: `0.055480`

## Interpretation

- This tests a repair path for the Stage43-CR failure mode by predicting a bounded residual around the safety floor instead of directly replacing the waypoint trajectory.
- The current heldout t100 policy remains unchanged unless a residual policy clears source/scene support and heldout safety gates.
- Future endpoints/full waypoints are labels only; inputs remain causal history, goal prototypes, baseline rollouts, floor rollout, domain/horizon tokens, and current state.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
