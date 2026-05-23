# M3W Model Card

- Architecture: JEPA-only, Transformer-only, and JEPA+Transformer hybrid compared in local-small.
- Outputs: expected baseline FDE, selected physical baseline, failure/interaction/occupancy/validity heads.
- No free residual correction, no Stage5C execution, no SMC.
- Deployment: fallback to Stage26 selector unless M3W gates beat it.
