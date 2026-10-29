# Model Card: Paired Completion Cost Heads

## Intended Use

Fitting-only diagnosis of seven-action cost-aware forecast selection. These
are 26,309-parameter cost heads, not new trajectory dynamics models. Neither
training mode earns deployment or independent confirmation.

## Inputs And Outputs

398 causal action/context fields, including 18 fixed static/past/rollout region
proxies. No future endpoint, future support mask, central velocity, test
endpoint goal or test-normalization statistic enters inference. The five
outputs are benefit, positive harm, reference, easy reference and easy harm;
completion mode trains conservative endpoints where outcomes are missing.
An estimated endpoint is not a calibrated upper/lower confidence bound.

## Training And Reproducibility

Six contexts, seeds 17/29/43, two candidate banks, two supervision modes:
72 fresh Torch fits at 2,000 updates. Normalizers come from physical fitting
data and are unchanged. Trajectory producer seed 43 is frozen. Controller
seeds do not constitute three end-to-end forecasting runs. Atomic checkpoints,
RNG/sampler/Adam recovery, a complete first refit and exact inference replays
are verified. Weights stay local; aggregate receipts and hashes are public.

## Performance And Limitations

Mixed known ADE gain: known-only +0.580799%, completion -18.512092%, fitting
only. Full required recording risk: 77/444 and 26/444. All scientific
advancement gates fail. Cost fit on a fixed region does not control the region
selected by the new head. Missing labels remain unknown; reference risk can
be undefined and is not replaced by zero. Additional training loss does not
establish a population guarantee.

No deployment change, Stage5C or SMC. No claim of independent generalization,
physical safety, true 3D, foundation capability, metric or effective seconds.
