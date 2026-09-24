# Pre-Readout Numerical and Provenance Repair

Independent read-only review identified three P2 defects before any new-policy
decisions or outcomes were calculated. None changes the registered training
labels, sampled rows, preprocessing, forecast predictions, model budget or rho.
The original source remains immutable because fitting checkpoints bind it.

1. Original solver tolerance could grow with an unselected large negative risk.
   A two-agent count-one example accepted risk1e-6 against budget1e-10. The guarded
   route checks selected risk with stable summation and NO allowance inflation.
   If rejected, queries with at most18 eligible agents use exhaustive original-unit
   optimization; larger rejected queries fail closed and are explicitly nonoptimal.
2. The guarded checkpoint reader requires requested view/action/seed and the
   corresponding frozen parent head, in addition to experiment/checkpoint hashes.
3. Aggregate replay refuses to run without a prior analysis and every outcome file.
   A first evaluation is not a replay. Decision replay already requires originals.

Training continues in net_easy_moment_v1 under its unchanged identity. The only
supported decision/readout route is run_m3w_net_easy_moment_guarded.py. It writes
separate guarded outputs and binds the original training identity and this repair.
The unguarded runner is retained solely for the registered training and provenance;
its decision/readout modes are not supported for reported results.

This repair was triggered by synthetic boundary tests and source inspection, not
new-policy outcome performance. No thresholds were tuned and no forecasts retrained.
All source roles and prior negative findings remain unchanged. Stage5C/SMC off.

A second pre-readout review found a NumPy-boolean JSON serialization defect in
the exhaustive branch and two verifier mismatches. The boolean is now native;
separate arithmetic uses the same strict stored-coefficient feasibility definition
and distinguishes explicitly nonoptimal floors from purported matched optima.
The initial guarded preflight identity is retained locally as superseded; it had
no fitted model, policy decisions or outcome evaluation. Original training identity
and checkpoints remain unchanged. The guarded identity is regenerated before the
first actual decision run, not after examining results.
