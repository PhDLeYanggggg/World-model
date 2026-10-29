# Prospective Separation of Policy Safety and Count-Matched Utility

## Authorization and Scope

On 29 October 2026 the user accepted all currently proposed decisions and
delegated routine auditing. This includes the immediately preceding proposal:
keep the original 2% full-policy safety requirement, use count-matched controls
for utility comparisons, mark empty risk comparisons not estimable, and retain
all old failed results. Audits remain required; user delegation is not permission
to suppress evidence, tune on test, reinterpret old results or execute Stage5C/SMC.
The project performs routine checks, repairs, experiments and safe Git updates
without repeated requests for the same authorization. Formal submission remains
subject to the existing final confirmation boundary.

## Observed Design Error

The fixed old rule required finite completion support for every full candidate
view and every count-matched candidate view. A zero-action anchor forces both
matched policies to select zero. Selected-harm/reference ratios then have zero
reference, so finite risk cannot be established, even for an otherwise safe full
candidate. Training longer cannot satisfy this requirement. This is a protocol
design error. Old registration, outputs and failed verdicts stay immutable.

## Future Rule

1. Freeze the complete view roster and all full/matched primary utility contrasts
   before any new evaluation. No comparator or zero-action view is dropped.
2. The actual full deployment candidate must satisfy the existing 2% all-risk,
   easy-risk and easy-preservation bounds in every required view, with finite
   completion support. Empty/unknown support remains a failure of advancement.
3. Count-matched contrasts test utility under equal interventions; they are not
   deployed policies. Their risk values are still reported, including undefined
   values, but are not an additional all-view finite-risk requirement.
4. Retain every original full/matched utility contrast and require its paired
   interval lower limit to be positive. Empty matched contributions are retained
   as zero utility where defined, not removed to inflate the aggregate. An
   undefined utility contrast does not pass. Risk is never imputed as zero.
5. Run structural preflight before admitting a new expensive experiment. Passing
   preflight proves only absence of the checked contradiction, not feasibility,
   model improvement, population safety or submission readiness.
6. This change is prospective. The previously observed unsafe scaled policy is
   not re-selected, re-scored or promoted under the new rule. All thresholds,
   data roles, unknown-outcome accounting and scientific claims remain unchanged.

The main protocol remains observation8/prediction12 in dataset-local coordinates,
with raw-frame stride12. Raw-frame t+50 is separate. The model remains 2.5D, not
true3D or foundation-scale; no metric/seconds or physical safety claim follows.
Stage5C execution and SMC remain prohibited. Independent roles remain unopened.
