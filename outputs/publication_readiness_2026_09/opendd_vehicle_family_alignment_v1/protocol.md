# Match the donor path family without changing the held action

The prior input-only audit found that the frozen incumbent uses
the learned path family for 98.34% of its recipient slots. Uniform
sampling over legal families at those slots would give 34.60%.
Test this discrepancy before changing the model or loss.

Replay each prior training draw with the same donor, recording,
query, recipients, seed, and initial family draw. Where the learned
family is legal, replace the drawn family with it; otherwise
preserve the original path. The matched-radius bank preserves
per-actor peak displacement across legal families. This experiment
changes training shapes, not held per-step displacements. Keep the
nested source-excluded forecasters and angle nominators frozen.
Train both existing screen losses for the same 200 updates on the
same number of examples. No model or threshold is selected on a
held score.

Compare each aligned screen directly with its frozen uniformly
sampled counterpart and with original action margin. All screens
use the same held actor and angle, original ego counts, and common
15% quota. Calibrate separate thresholds on the opposite excluded
location under the same 2% risk, two references and positive
incremental utility checks. Report uncertainty by seven paired
location means, after averaging directions and seeds.

Report the realized training family mix, the number of changed
exact nominees, and strict label frequency. This is a family
alignment ablation, not a claim to have matched the incumbent's
actor-selection distribution. OpenDD remains development; no new
external outcomes, release promotion, Stage5C or SMC.
