# Model and Data Card

- Model: source-balanced ridge regression,380 existing causal features, five
  moment outputs for each of two target references.108 joint fits,216 heads.
- Preprocessing: means/std and cost scale from known training rows in the two
  declared probe-fitting localities only. Feature clip10; regularization0.01.
- Forecasts: nine previously trained unit-repaired neural banks, cached_verified.
- Floor: frozen protected damping/CV policy; forecaster/floor fitting sources
  disjoint from all new probe rows. Same checkpoints serve fitting/readout.
- Features: current/past motion and neighbor geometry, causal baseline/neural
  rollouts and floor action. No future labels, availability masks or errors as
  inputs. Future observed positions supply training labels and evaluation only.
- Targets: positive benefit, positive harm, reference error, easy-event reference
  and easy-event harm. Easy event stays defined by producer-training CV cuts.
- Protocol: twelve opened European source localities; four/four/two/two role
  separation; all six probe fit/held pairs; forecaster seeds17/29/43.
- Data: released detector tracks, image-local coordinates, obs8/pred12 at raw
  stride12. Silver, not human gold. Overlapping windows are not independent.
- Safety: predicted screened policy preserves observed net easy ADE, but fails
  positive-harm budget. `safe` in an action name is not a certified-safe label.
- Intended use: opened-source method development and failure diagnosis only.
- Prohibited claims: independent confirmation, deployment readiness, verified
  metric/time, historical Stage37 recertification, physical safety, true3D or
  foundation model. No Stage5C execution or SMC.
- Limitations: weak linear head, two fitting localities per view, source imbalance
  despite equal-source weights, silver/missing labels, no new scene/interaction
  contribution ablation. All bounds concern prediction disagreement, not risk.
