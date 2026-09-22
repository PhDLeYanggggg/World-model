# EqMotion Pair-Excluded Training and Cost Data: Verified

Completed 2026-09-22. This is an upstream training-data prerequisite, not a new
selection-policy result. The eighteen fresh EqMotion fits all complete the fixed
4,000 updates, totaling72,000updates and4,608,000draws. Recorded fitting time is
27,903.47seconds (7.75hours). Every checkpoint matches its Transformer reference
in fitting row IDs, loss factors and row-level draws; excluded-site draws are zero.

Six pairs of excluded physical sites times three seeds supply12 outer head views.
For a head evaluated on site A, the predictor producing a fitting row from B was
trained and normalized on neither A nor B. Eighteen private prediction archives
contain1,581,804repeated query/pair/seed rows, drawn from175,756unique eligible
queries across four source sites. These repeated rows are not independent samples.
Prediction archives contain IDs and predictions only; future costs remain in
separate supervision archives. The25,191unknown-cost instances remain unknown.

## Verification

- All18models and12cost views are complete; the cache process exited0.
- Fixed first/middle/last128-row blocks replay exactly:5,484rows across18models.
  This is fixed-block replay, not full-cache checkpoint replay.
- Separate arithmetic verifies1,581,804cost rows,36ordered exclusion relations,
  12view memberships, all18samplers and unknown-label preservation. This is a
  separately implemented same-agent check, not independent research replication.
- The related upstream and new downstream preparation suite passes89tests.
  Tiny training tests are engineering evidence, not substitutes for real fits.
- Analysis SHA256:
  `8367e7bd5628e01fa04d1c5451ec2c8816ddbee60dac07b1cd36a16b0eb05f77`.

The first cache process was interrupted after seven complete producer archives.
The old PID was confirmed absent before resuming. All seven complete archives
were reused; the unfinished producer was regenerated. No fitting was repeated,
budget reduced, threshold changed or result-selected restart introduced.

## Evidence and Next Step

`fresh_run`: eighteen fits, new caches, fixed-block replay and arithmetic audit.
`cached_verified`: twelve outer EqMotion forecasters, eighteen matched Transformer
sampling references and hash-bound causal source inputs.
`not_run` at completion of this prerequisite: new cost-head fitting and readout,
independent calibration/confirmation and deployment. The next registered study
refits36cost heads and compares them with the failed frozen transfer; its outcome
is not implied by completing these data.

All four SDD sites are research-design exposed despite per-fit exclusion. The
task remains eight observed/twelve predicted sampled annotation steps, SDD
stride12, annotation pixels. No verified seconds, metric scale, true3D, foundation
success, new deployment, Stage5C or SMC. Independent generalization and a useful
methodological contribution remain unproven.
