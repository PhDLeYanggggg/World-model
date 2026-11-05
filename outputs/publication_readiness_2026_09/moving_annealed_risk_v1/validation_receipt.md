# Validation Receipt

Observed on 5 November 2026 using native arm64 Python 3.11.1 and Torch 2.12.0.
Actual training used CPU4, inter-op1, workers0; no resource probing or fallback
simulation of neural updates.

- Pilot PID90244: exit0, 35.7312s, 400 actual updates including repeats.
- Training PID90371: exit0, 494.9384s, peak RSS7,561,248,768bytes;
  six primary fits / 12,000updates plus an exact 2,000-update full refit.
- Evaluation PID91232: exit0, 47.3703s.
- Verification PID91354: exit0, 46.3732s; all3prediction/action/metric blocks
  match exactly; old cached-control metrics are recomputed exactly.
- Report generation and report replay: both observed exit0. All26existing
  public evidence files have identical SHA256 hashes before/after report replay.
  This receipt was added after that comparison.
- State gain, contrast, completion bounds, worst easy-net, violation counts,
  support counts and expansion result match the analysis exactly.
- Analysis SHA256:
  `b20a3882c1f96eb6bae233196043c7293fc74c770525cbb2c68de67aa0f4347b`.
- No remaining experiment process was observed after completion.

Focused tests: **34 passed in 4.03s** across annealed-risk training/reporting,
all/easy diagnostic, signed-decision, direct-tail risk, kinematic risk,
frozen-readout, admission-priority and forecast-readout tests. Tests cover exact
resume, exact old constant updates, future-field rejection, immutable schedule
identity and failure-preserving reporting. The full historical test suite was
not rerun. No changed core training code followed these checks; later edits
only clarified report prose and research-state results.

These checks establish the tested implementation and replay, not scientific
success. Original safety fails, deployment remains unchanged and no independent
confirmation was performed. No identity certificate, Stage5C execution or SMC.
