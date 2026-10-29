# Protected Risk Estimation Versus Action-Coupled Continuation

## Material Passport

Mode: run and validate a single matched objective repair. Previous turn:
progress, completed fixed-cohort diagnosis with preserved negative evidence.
Native arm64 local CPU4/inter-op1/workers0. CREATE supplies only hash-verified
original-TRAIN packets; no remote training, authentication change or new data role.
Original obs8/pred12 rawstride12, source split, detector-silver labels and2%
positive selected harm/reference criteria remain unchanged. No deployment,
Stage5C or SMC. No metric/seconds/physical-safety/true3D/foundation claim.

## Hypothesis and Fixed Arms

Current risk heads fit a scalar action objective without reliably estimating
easy harm on selected rows. Holding benefit/reference predictions fixed,
removing decision-regret and dual action gradients may improve cost estimation
without simply spending more updates or changing action count. It may fail
because the decoder remains poorly conditioned or causal features insufficient.

Take all18 final self heads:6contexts x3headseeds17/29/43. Duplicate each exact
model, optimizer moments, step counter and source sampler state:

- coupled: continue primary moment/signed-score loss + decision regret + original
  source-scaled risk dual terms, with original dual updates;
- estimation: continue only the exact primary moment/signed-score loss; unused
  risk duals remain frozen. This removes both action-dependent terms as a bundle.

Both get2,000 additional updates. No sweep, early stopping or best-epoch selection.
Thirty-six new continuations,72,000 real optimizer steps. The rational decoder,
features, harm encoder, clip5, learning rate/weight decay and source-query draw
budget stay fixed. Final temperature remains0.1. The original frozen self is a
third cached control, not an equally trained new arm. Benefit B, reference R
and easy reference R_easy must remain bit-exact across all three predictions.
No added scene context or feature, normalization refit, teacher, forecast or
trajectory residual. All source packets use the original whole-recording fit
partition; discard internal held rows before any forward in this experiment.

Continuation retains prior optimizer history on purpose. This isolates a local
repair from an already action-trained state, not superiority from scratch.
All unknown-label rows remain in causal input/context and inference; supervised
draws use only known rows. No future mask, target or endpoint enters inference.

## Pilot, Freeze and Readout

First100 updates per arm; estimation50+50 interrupt/resume must match a separate
direct100-update run exactly. Verify copied risk encoder updates, protected
outputs, input hashes and matched sampled rows. This is300 actual pilot updates,
not evidence of scientific lift. Retain checkpoints, heartbeat and PID. Proceed
with all36 full continuations only after engineering pilot succeeds.

Commit code/protocol/registration before pilot and training. Freeze all36 finals
and commit their receipts before fixed-cohort inference. No held readout in this
protocol. Evaluate all72 fitting context/source/seed views against original
self-selected and all eligible cohorts. Match the prior original predictions
and cohort hashes exactly. Outcomes define supervised errors, never the cohort.

Report all/easy signed-risk MSE, positive harm masses, unknown support, full
policies, known/upper selected risk, empty/undefined coverage, full and matched-
count completion-lower utility. MSE is source-normalized and equal-view averaged;
not raw trajectory ADE/FDE. Repeated occurrences and views are not independent
samples. Fitting comparisons receive no confirmatory CI or significance claim.

## Prospective Fitting Routing Screen

Estimation is eligible for a separately designed later held comparison only if:
1. On the frozen original-self cohort, equal-view easy signed-risk MSE is strictly
   lower than BOTH original and coupled (absolute numerical slack1e-12).
2. All signed-risk MSE on that cohort is no worse than either comparator.
3. All/easy signed-risk MSE on all eligible rows is no worse than either.
4. At least12/18 context/head-seed pairs improve original-self easy MSE versus
   BOTH comparators, averaging four sources within pair.

All nine Boolean conditions are retained. No failed source/seed/cohort is removed.
An empty known cohort makes its MSE undefined and fails the relevant condition.
This is an internal fitting gate, not a changed scientific risk tolerance or
deployment pass. Even a pass authorizes no independent readout; any later held
design must be frozen separately. If it fails, do not run another held readout.
Do not choose a favorable checkpoint or threshold after results.

## Operations and Reproducibility

Single process, atomic checkpoint100, heartbeat100, explicit --resume only for
incomplete runs.12h recovery boundary,10GiB free reserve,256MiB private model
cap. Keep v1/failures/old artifacts. No raw/cache/checkpoints/images in Git.
Tests include exact resume, unchanged origin/utility, objective replay, sampler
matching, unknown-row accounting and incomplete-roster rejection. Registration
and model hashes bind each phase. The resulting model is a risk-estimation
research candidate only; the long-term paper/world-model goal remains active.
