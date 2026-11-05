# Failure Analysis: Learned Risk Differences Are Not Yet Reliable

## What Was Changed

The preceding attribution separated causal history, neighbor, forecast and
forecast/history differences in fixed cross-source matches. Harmful source076
matches had more lateral forecast/history mismatch, while other sources had
different patterns. I tested a model-side response: learn f(x) on donor sources
and predict f(query) plus the weighted donor residual y-f(donor). Reference IDs,
source/query weights, signed targets, original 120 features and risk rules stayed
fixed. The plain f(query) and fixed ridge transport were retained as controls.

This is real Torch risk-regression training, not trajectory correction, latent
generation or a new world-dynamics success. Eight networks and eight ridge
models complete; all fitting rows receive 32 epochs. No new C/H was opened.

## What Improved

Transport reduces source-macro decision cost relative to the same plain neural
model by 1.769447%, 6.433618%, 0.892890%. This supports a small conditional
adjustment effect within this exposed development comparison, not a safe
deployment or statistically confirmed contribution. The source076 context
also improves substantially relative to naive track32 retrieval.

## Why It Still Fails

1. **The primary comparison itself is negative.** Relative to shared64,
   decision-cost reductions are -6.914305%, -54.521863%, -5.517107%. Failure is
   not merely an artifact of the extra no-increased-harm screening clause:
   costs are worse in every context before that clause is considered.
2. **Harmful admissions remain excessive.** Counts change 0/1/0 to 94/353/1288.
   Compared with the plain neural control, transport also admits more harmful
   cases, despite a modest improvement in total severity-weighted decision cost.
   The original 2% full-policy requirement is not demonstrated by either branch.
3. **Source-conditional transfer is weak.** Plain neural source076 normalized
   held MSE is 7.738072 versus 0.525897 fitting MSE. Every fold has larger source-
   held than fitting MSE. This is consistent with source mismatch and generalization
   error; it does not isolate a unique causal failure or prove irreducibility.
4. **Optimization completion is not convergence.** All registered 32 epochs ran,
   but source076 has 716 donor rows and only 64 updates. Other small folds get 128-256
   updates; larger folds get1,248-1,888. Last-eight-epoch fitting loss still
   falls in all folds. The experiment does not establish that the architecture
   was optimized to convergence, nor that more fitting will improve source-held risk.
5. **Residuals are in-sample.** Donor outcomes were used to fit f and then to
   form its residuals. There is no query-source leakage, but donor residuals
   may be optimistic and their local average may not transfer. This is not
   independent residual calibration or a conformal guarantee.
6. **Selected support is unresolved.** Source053 admitted proposals have no
   known easy reference; source065 has no admissions. Their selected easy
   ratios stay NA. The component experiment does not repair the original
   all-view support problem, and no identity exemption is introduced.

## Controls And Interpretation

The fixed ridge transport has a small +0.325424% cost reduction in context 1,
but 92 harmful admissions versus the forest's 1. It was secondary, is below the
fixed 1% information threshold, and cannot become the primary posthoc. In
context 2, neural transport lowers MSE 32.192% while worsening decision cost.
Mean squared risk prediction and safe decision quality are different objectives.

These findings do not justify another source-held threshold sweep or a claim
that deeper networks are necessarily better. The original 2% rules remain.
The next controlled check should hold architecture/features/reference IDs fixed
and examine a predeclared longer fitting budget with optimizer recovery, using
training loss for convergence evidence rather than selecting epochs on the
already-exposed source-held outcomes. Preserve this 32-epoch failure. If fitting
improves but transfer does not, investigate residual independence and source
context rather than widening the same model. This follow-up is not_run here.

## Remaining Paper Gap

The original 24-group primary remains NA, and the all-view support contract has
not passed. There is no new independent confirmation, formal three-seed main
result or safe full-policy improvement. This module cannot serve as a successful
main-method claim. Deployment is unchanged; the paper is not submission-ready.
Native 8/12 silver image-coordinate limits remain. Stage5C/SMC remain disabled.
