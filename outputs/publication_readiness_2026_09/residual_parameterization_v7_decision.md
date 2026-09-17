# Frozen Development Decision: Baseline-Relative Output Parameterization

Date: 2026-09-17. Authorized under the user's explicit delegation to pursue
the obs8/pred12 research route. This is adaptive development after negative v6
results, not preregistration of independent confirmation.

## Hypothesis

v6 completed both model families and every seed. Near-stationary past histories
account for most additional neural error, but that is descriptive evidence,
not proof of the cause. We test whether an explicit CV residual parameterization
and a motion-dependent amplitude bound reduce avoidable drift while retaining
useful changes on moving agents. A purely stationary observed history may be
followed by movement; failures to predict that movement must remain scored.

## Two Arms, Both Fully Trained

1. `baseline_skip`: the v6 conditioned Transformer core is interpreted as a
   residual, added to the declared CV rollout. Its last affine layer starts at
   zero, so the initial forecast is exactly CV. Later residuals are unbounded.
2. `motion_bounded`: identical core, initialization, data, loss, optimizer and
   sampling. Only the output residual mapping changes. For restored residual
   d and requested fraction u, output is B + A*u*d/sqrt(1+||d||^2), where A is
   max(observed ego path length, maximum requested CV displacement), measured
   in the unchanged past-normalized coordinates. A contains no future label,
   neighbor-distance scale or learned test statistic. No trainable parameter
   is added. The radius depends on observed motion, not a target-defined easy label.

The implementation evaluates the radial map with numerically stable rescaling;
its derivative at zero is one, so zero initialization does not deadlock learning.
The bound limits correction magnitude, not realized prediction harm or collision
risk. It cannot produce a physical or statistical safety certificate.

The new skip versus bounded comparison isolates the amplitude mapping. Comparing
either to cached v6 also changes baseline reparameterization/initialization and
must not be described as an amplitude-only ablation. The completed v6 results
and code remain at commit `052bcc64`; no old checkpoint identity is rewritten.

## Unchanged Scientific Contract

- Primary: eight observed/twelve predicted native annotation steps, past-normalized
  ADE, equal physical-scene aggregation. Raw-frame t+50 stays supplemental.
- Same fit/development source hashes, 11,966 fit windows, continuous Students01
  population, complete-aligned observed neighbors and exact target timestamps.
- Same three physical producer folds, seeds 17/29/43, CV floor, 10,000 updates per
  full/held-fold forecaster, batch32, Smooth-L1, lr0.0003 and gradient clipping.
- Same OOF ridge and 1,000-update neural gain/harm heads, causal feature schema,
  easy/hard quartiles, both fixed policies and the <=2% easy selection ceiling.
- No sample exclusion, new scale floor, new threshold search or future-validity
  information in inputs. Labels are opened only after inference payloads exist.
- All arms and seeds are reported; a negative seed is not dropped. Frozen-budget
  checkpoints are used, not the best training minibatch or a development-selected
  epoch. Selection is development-only and cannot establish independent success.

## Runtime and Evaluation

Use arm64 CPU4/interop1/workers0. Run a 100-step fit-only pilot for each arm,
then resume those exact checkpoints into the full sequence. Keep logs, losses,
atomic checkpoints and heartbeat. A healthy local run within the user's time
allowance is not shortened. No new CREATE task is needed for this local scale;
remote authorization status has not changed.

Primary comparisons include CV, uncontrolled network, independent intervention,
scene-uniform, joint selection, both OOF heads and both frozen policies. Retain
native per-recording causal-baseline context, absolute/relative easy damage,
hard error, candidate/CV oracle, coverage and compute. Do not choose another
metric after seeing which arm improves it. Source-clock, homography and source
annotation-generation caveats remain unchanged. One development site cannot
support a useful scene CI; report three-seed variation without renaming it.

Success for this development hypothesis requires improved prediction with easy
preservation under the original selection rule, not merely lower training loss
or exact CV copying. Genuine paper claims still require independent scenes and
strong deferral/matched-count controls. No deployment upgrade, Stage5C execution,
SMC, true-3D, metric/seconds, foundation or submission-ready claim is authorized.
