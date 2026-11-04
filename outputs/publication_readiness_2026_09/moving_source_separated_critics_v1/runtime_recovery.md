# Interrupted Training: Time Accounting

The original PID89603 and session34785 were confirmed absent at2026-11-04T22:21:44Z. No exit code or cause was observed.71complete models and the step100partial state were retained. Same-binding resume continued the saved optimizer/sampler/RNG; no model, normalizer, risk rule, threshold or update budget changed.

The train-execution receipt describes only terminal resumed PID21038 and its2760.31s elapsed time. It is NOT the full interrupted training wall span. Sum of saved primary fitter times:8822.70s. The original last-heartbeat elapsed time is6440.0074s, not an exact terminal duration. Do not sum overlapping timing definitions or count the unobserved wall gap as compute. Pilot1200and first full-refit2000updates are extra. Completed primary checkpoints contain exactly216000prescribed updates; any unsaved work before termination is unknown, not asserted as zero.
