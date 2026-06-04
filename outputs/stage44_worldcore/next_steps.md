# Stage44 Next Steps

- If no-baseline remains weak, add longer K=64/128 history and source-balanced normalization.
- If JEPA lift is weak, split future world-state targets into explicit trajectory, interaction, occupancy, and goal-route latents.
- If scene/interaction ablations are negative, replace proxy vectors with dynamic graph tokens and audited scene packs before claiming multimodal contribution.
- Keep Stage5C and SMC disabled until protected latent dynamics passes deployment gates.
