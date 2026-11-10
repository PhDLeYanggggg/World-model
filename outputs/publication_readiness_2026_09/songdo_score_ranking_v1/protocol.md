# Diagnose the frozen Songdo release score

Use the already published scalar replay of the 40 calibration flights and
160 confirmation flights. Do not read trajectory labels, fit a score,
choose a threshold, or change the original operating rule.

On calibration, for each of the four proposal rules and three seeds,
recover the original supported high-score prefix. Describe the first
blocking score group (including ties): missing risk denominator only,
measured violation only, both, or supported. Count supported and dual-pass
proposals elsewhere on the same roster. A missing denominator is not a
measured violation unless another defined risk ratio also exceeds 2%.

On confirmation, isolate ranking from the proposed forecasts. For each
rule, within each linked source component and seed, take the top
`max(1, floor(n/4))` scores. At the cutoff give tied proposals their equal
selection probabilities. Compare against the exact expectation of uniform
selection of the same number of that rule's proposals. Keep all components,
including any for which the whole roster fits within the quota. Report the
selected proportions for dual pass (primary), original risk pass, undefined
risk and measured violations. Unknown risk is never a pass. Average seeds
within component, then weight components equally. Reuse the original 3,000
paired component draws for 95% percentile intervals. Report all four rules;
the primary contrast is frozen joint score minus random, not whichever
result looks strongest. This is a retrospective ranking diagnostic that uses
the full score list, not a releasable online policy or independent new test.
