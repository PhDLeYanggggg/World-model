# World Model V3 Target Spec

## Principle

The target is not a prettier trajectory predictor. The target is a falsifiable, camera-aware, physically constrained, multi-agent state-space world model that can say:

1. what is observed,
2. what is inferred,
3. what is uncertain,
4. what physical rules were applied,
5. what was learned from data,
6. what future modes are plausible,
7. and why a prediction should not be trusted.

If the model cannot fail loudly, it is not a serious world model.

## Current Expert Verdict

Current Stage 2 audit:

```text
score = 49 / 100
verdict = prototype_with_major_failures
```

This is not yet an exceptional world model.

It is a useful pseudo-3D / 2.5D learned-residual scaffold with verified synthetic t+100 evaluation. It fails the world-class bar on:

- long-horizon FDE,
- learned dynamics margin over hand physics,
- SMC coverage,
- terminal semantic diversity,
- real long-trajectory validation.

## V3 Architecture

### 1. State-Space Core

Each agent state:

```text
S_i(t) = {
  X, Y, Z_ground,
  Vx, Vy,
  Ax, Ay,
  radius,
  height,
  desired_speed,
  goal_distribution,
  intent_latent,
  group_latent,
  uncertainty_covariance
}
```

No image observation may be treated as true state. Image observations are measurements.

### 2. Observation / Filtering

Required:

```text
p(O_t | S_t, camera, scene)
```

For synthetic:

- true state is available,
- observation noise can be simulated,
- filtering can be evaluated directly.

For real data:

- AerialMPT bauma3 only verifies to t+12,
- t+100 remains qualitative until a longer real sequence is loaded.

### 3. Learned Dynamics

V3 must not rely on one-step residual loss alone.

Training objective:

```text
L = one_step_state_loss
  + rollout_loss@{10,25,50,100}
  + physical_violation_loss
  + goal_posterior_loss
  + stochastic_coverage_loss
  + calibration_loss
```

The learned model must beat hand physics by at least:

```text
ADE@100 improvement >= 10%
FDE@100 improvement >= 10%
```

Otherwise it remains only a useful residual experiment.

### 4. Goal / Intent Posterior

SMC particles must not be mere Gaussian noise.

Particle:

```text
particle = {
  world_state,
  latent_goal_distribution,
  latent_intent,
  body_params,
  log_weight,
  trajectory_history,
  diagnostics
}
```

Goal proposal:

```text
q(g_i | history_i, scene, exits, flow, group)
```

A valid future mode should correspond to a different causal/intent hypothesis:

- different exit,
- detour versus direct,
- jammed versus free-flow,
- split versus merge,
- stop versus continue.

### 5. Physical Constraint Layer

Projection is allowed, but projection must not hide bad proposals.

For every rollout:

```text
state_corrected, projection_cost = project_constraints(state)
log_weight -= lambda_projection * projection_cost
```

Report both:

- post-projection violation,
- pre-projection violation / projection cost.

### 6. Scene Graph

The model must use:

- walkable polygon,
- obstacle polygons,
- exits,
- corridor bottlenecks,
- density regions,
- flow priors.

No long-horizon result should be trusted without scene geometry.

### 7. Multibranch Prediction

Minimum serious benchmark:

```text
particles >= 64
coverage@64 > 0.15
best_of_64_FDE@100 < 5m synthetic
semantic_cluster_diversity > 0.55
semantic_event_accuracy > 0.40
```

If these fail, the model must say the branch predictor is not yet useful.

## Experiment Protocol

### Synthetic

SyntheticPhysicalCrowd2.5D must include:

- open passage,
- corridor jam,
- obstacle detour,
- crossing flows,
- group split,
- group merge,
- temporary stops,
- sudden goal changes,
- near-collision avoidance.

Required metrics:

- ADE/FDE @ 1, 10, 25, 50, 100,
- best-of-64 ADE/FDE,
- coverage@64,
- collision rate,
- obstacle rate,
- boundary rate,
- speed/acceleration violation,
- projection cost,
- semantic event accuracy,
- cluster diversity,
- calibration/NLL.

### Real Data

V3 cannot claim real t+100 until one of these is connected:

- Stanford Drone Dataset,
- TrajNet++,
- ETH/UCY,
- a longer AerialMPT sequence with at least 100 future frames.

For AerialMPT bauma3 current slice:

```text
verified horizon <= t+12
t+100 = qualitative free-run only
```

## Red Lines

The system must fail its own audit if:

1. real t+100 metrics are reported without real t+100 labels,
2. learned dynamics does not beat hand physics meaningfully,
3. branch count is less than the stated best-of-N,
4. coverage remains zero,
5. terminal clusters collapse to one semantic family,
6. boundary/collision rates are hidden by projection,
7. camera calibration uncertainty is omitted.

## Next Build Order

1. Stabilize runtime so full `run_stage2_demo.py` can run after latent-goal SMC changes.
2. Add pre-projection violation and projection-cost metrics.
3. Train with multi-step rollout loss.
4. Add learned goal posterior.
5. Run true 64-particle evaluation.
6. Connect a real long-trajectory dataset.
7. Re-run `python run_world_model_audit.py`; do not claim progress unless score improves.
