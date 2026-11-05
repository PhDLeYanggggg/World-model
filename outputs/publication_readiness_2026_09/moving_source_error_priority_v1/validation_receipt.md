# Completion Validation

- Real discovery pilot PID80481: observed exit0; exact resumed predictions.
- Teacher process PID80596: observed exit0; 8/8 complete, full refit exact.
- Student process PID81750: observed exit0; 6/6 complete, full refit exact.
- Evaluation PID82614 and replay PID82755: observed exit0, all three blocks exact.
- Reporter and its fresh rerun: observed exit0. All 44 existing public artifact
  hashes remain unchanged by the report rerun.
- Actual AdamW step counts, finite trained weights, encoder changes, matched
  student inputs and initialization, and discovery-mask bindings were checked.
- Research-state main metrics match saved analysis exactly.
- Focused tests: 21 passed in 2.39s. Full historical suite not rerun.
- No public artifact in this study exceeds1MiB. Private checkpoint/discovery
  paths remain Git-ignored. No raw data, private cache or model weights added.
- Unrelated staged-work sentinel remains unchanged before final integration.

The first pilot's cache-interface failure and its pre-training repair remain
in the preflight amendment. A local checksum command initially failed under
an unavailable C.UTF-8 locale; rerunning with LC_ALL=C/LANG=C succeeded. That
read-only tool failure changed no artifact, model, rule or result.

Analysis SHA256:
`bd3635dc075b14a9b1ca72ff1934563aeb3f59db80691af4107d44c6e87a1631`.

Original scientific gates fail. No identity certificate, deployment upgrade,
new independent test, Stage5C execution or SMC activation.
