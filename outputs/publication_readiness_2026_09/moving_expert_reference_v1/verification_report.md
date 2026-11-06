# Actual Execution And Verification

Native arm64 CPU4/inter-op1/workers0. All executions below have observed exit0;
no new CREATE job, resource probing, multiprocessing or NumPy training fallback.

| Phase | PID | Seconds | Peak RSS bytes | Result |
|---|---:|---:|---:|---|
| Context preparation | 82185 | 74.705404 | 2928934912 | Five exact ID/feature joins |
| Actual resume pilot | 82366 | 46.005963 | 3850371072 | 128+128 equals continuous256 |
| Primary fitting | 82450 | 483.161025 | 7010271232 | 48models,196608updates |
| Complete refitting | 83520 | 496.797578 | 6949060608 | All48 model/optimizer/trace states exact |
| Complete causal inference | 84257 | 192.355692 | 2054864896 | All18 new caches frozen before readout |
| Original-contract evaluation | 84742 | 5.445373 | 2071691264 | Frozen prior controls also exact |
| Actual fixed-result replay | 85082 | 17.827980 | 2552086528 | Predictions, choices/checks and metrics reproduce |

The pilot's274.244s pure-fit projection excluded I/O, repeated normalization,
full refits and allocation. Actual primary runtime is483.161s, not that estimate.
Healthy processes were not stopped or downscaled because of speed. Checkpoints,
heartbeats, unique execution records and explicit resume remain available.

## Scientific Definition And Engineering Scope

The experiment was registered/committed before actual preparation. No model,
protocol, training settings or scientific threshold changed afterward. Both
arms have identical initialization and draws. All24initial minibatch losses
match; context lowers all24final paired training minibatches, which is a fitting
manipulation check, not held-source proof. H/EH and support remain bitwise exact
per candidate. Reference is broadcast once per agent-window, not per action.

The unchanged allocator processes6,522repeated query views for each joint mode.
There is one free-joint and four matched-joint postcheck failures; each retains
the feasible independent parent as preregistered. They are nonoptimal fallbacks,
not dropped rows or certified global optima. Native HiGHS emitted an internal
`transformNewIntegerFeasibleSolution` diagnostic on stdout; execution completed
normally and the allocator's reason counts retain these numerical safeguards.
No runtime repair or relaxed risk tolerance was applied.

## Replay And Tests

Final second replay PID87047 also exits0 in17.673682s,peak2,629,582,848B after
the report/diagnostic writers finish. All predictions, choice checks and metrics
again agree exactly, with public files unchanged during that check. Its unique
private receipt is retained alongside PID85082; neither reruns timed search.

Actual replay recomputes all reference predictions exactly, checks proposal and
independent choices exactly, verifies cached uniform/joint feasibility and
per-query matched counts, and recomputes the full metrics exactly. Public files
must remain hash-identical during the check. Its unique receipt is private under
`data/stage_cvpr2027_experiments/moving_expert_reference_v1/replays/`.
Timed mixed-integer search is not rerun; feasible cached choices are not a claim
of deterministic search or independent scientific confirmation.

The final eleven-file focused suite passes35tests in2.65s, covering the new
context join, zero-initialized treatment, source-excluded normalization, actual
optimizer recovery, frozen harm/support, action invariance, original decisions,
empty-control NA, diagnostics and report gates. The full historical suite was
not rerun. These are engineering checks; actual scientific outcomes remain in
analysis.json and fail the complete candidate screen.

The two-panel local figure was visually inspected: all policy controls fit and
the unmet72-view requirement is explicit. No private arrays, weights, images,
videos or third-party source files are published. The compiled main paper is
not modified by this experiment. New independent confirmation and three teacher
seeds remain not_run; original24-source-group main remains NA. Stage5C/SMC off.

The separately generated real failure-case figure was also inspected. It uses
only8available labels out of12requested, with gaps left unfilled. It is a posthoc
illustration, not a representative estimate or a complete12future error claim.
Raw paths and both figures remain private; only scalar summaries are public.
