# What this control can and cannot establish

## Material Passport

This is a matched SOURCE-fitting neural risk-head experiment,not a novel
trajectory model or an independent result. Prior inputs/forecast heads are
cached_verified. The real pilot is fresh_run;full fitting and policy outcomes
are not claimed by this design note. The registered primary is unchanged.

## Controlled Parametrization

Let B0,H0,R0,ER0,EH0 be the original predicted moments and C the original
causal error envelope. The protected head changes H/EH but freezes B/R/ER.
Its allowed harm capacity is max(C-B0,H0),including original anchor roundoff.
The released head can adjust B,then remap the learned harm fraction into
the remaining capacity. References R0/ER0 remain identical in both arms.
Both begin exactly at the same predictions and have the same30,857 parameters.
The control's unused benefit branch cannot affect its training gradients.

True observed risk still uses the original evaluation targets and denominators.
Coupled predicted moments neither redefine that risk nor certify calibration.
Reducing predicted H by increasing predicted B may make model utility look
better while worsening actual decisions;complete observed risk checks must
catch this. A lower fitting MSE is not sufficient to advance or deploy.

## Relation To Primary Literature

[Liu and Grigas,NeurIPS2021](https://papers.neurips.cc/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf)
study surrogate-to-decision risk calibration for predict-then-optimize with a
linear objective and a known convex feasible set. Here predicted risk moments
also affect feasibility and agent decisions are discrete. Their theory does
not certify this joint allocation or our squared-moment objective. The relevant
lesson is to measure downstream decisions,not infer them from regression loss.

[Angelopoulos et al.,ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
give expected-risk control for exchangeable,monotone bounded loss functions.
An intervention-conditioned positive-harm/reference ratio need not be monotone
when fewer agents switch;overlapping windows are not independent calibration
examples. Those assumptions are not established by the current SOURCE fitting
experiment. We claim neither conformal control nor a population guarantee.

These ordinary moment and envelope constraints are not,by themselves,a novel
method contribution. A useful M3W claim still needs a demonstrated neural
increment,non-destructive decisions,strong task-matched controls and independent
scene evidence. Stage5C and SMC remain closed.
