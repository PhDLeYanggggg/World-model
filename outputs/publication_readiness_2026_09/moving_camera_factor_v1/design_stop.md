# One-Sided Operator Design Stopped Before Readout

The registered experiment is **incomplete**,not a completed negative performance
result. Its full identity control and real recovery pilot remain valid. No new
donor outcome,forecast bank,primary statistic or component pass was computed.
Only the first past_only fit reached2160updates;the other eleven did not run.
The4000update target is not relabelled complete or downgraded to a quick result.
The original protocol/code/registration remain unchanged in Git.

## Structural Input Defect

A synthetic counterexample uses two constant translating cameras and the same
world-coordinate pedestrian motion. Let native history be h_t=p_t+c_t. Exact
past alignment to the query produces h'_t=p_t+c_0. After current-origin removal,
the network's geometric history is p_t-p_0 for both camera motions. Its native
future label remains(p_future-p_0)+(c_future-c_0),which differs between cases.
The core recomputes its baseline from aligned history;it receives no separate
raw history or camera tokens. A deterministic function of identical core inputs
cannot predict both different native labels. This is a conventional information-
loss counterexample,not a new general theorem or a real-data score.

Two new synthetic checks pass1.10s. Actual small pinned-core outputs are exactly
the same for the two past_only cases,despite different labels. The both arm
retains the expected camera-dependent difference at output. This does not
prove identical-input collisions occur in the real dataset or identify every
cause of the previous both-arm failure. It shows that disabling one side of a
coordinate change is not a valid general-purpose repair without preserving
the discarded view information. No real future/test endpoint enters the test.

## Explicit Stop,Not Slowness Or Deadlock

At2026-11-06T23:20:31Z,the active processPID78512was intentionally sentSIGTERM
to repair this model design,not because training was slow. Observed exit130.
Its handler wrote a step2160checkpoint and interruption heartbeat. Model files
are retained;no data or valid full control is removed. No current model is
deployed. New model definitions must register separately before fitting.

The signal arrived inside a forward pass. Although the counter and draws record
2160committed updates,the sampler may already include an incomplete next-batch
draw. This checkpoint is **not asserted to be a coherent exact-resume boundary**
and is not reused for the next model. The earlier recovery pilot checked
completed-update boundaries;it did not prove mid-update signal atomicity.
The next training wrapper must defer stop requests to committed-update
boundaries and avoid overwriting a coherent checkpoint after a partial update.

## Actual Repair Direction

Keep original native history/neighbor coordinates as the core input. Supply
aligned history as additional per-node past information,not a replacement.
Do not force a future camera extrapolation onto the output. A matched native-
kinematic capacity control is needed to distinguish camera information from
additional parameters. The original whole12loss,source exclusions,budgets,
primary-risk2%/support/NA rules and no-new-test boundary remain. This direction
is not yet trained or a claimed success. Stage5C execution and SMC remain off.
