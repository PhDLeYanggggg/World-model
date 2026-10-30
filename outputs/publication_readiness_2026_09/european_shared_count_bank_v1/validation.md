# Execution and validation receipt

## Real scientific computation

- Pilot PID31533 exits0 in22.831885s, peak RSS9,772,220,416B. Its128 real
  queries and3267rows reproduce exactly after a deliberate interruption.
  No outcome scoring occurs in this engineering pilot.
- Complete readout PID32631 exits0 in724.190102s, peak RSS9,810,214,912B.
  All72 groups, six contexts and cost-head seeds17/29/43 are included.
- Separate full replay PID34909 exits0 in753.266814s, peak RSS7,615,741,952B.
  All72 new inference/allocation/score groups and144 legacy full-policy
  controls agree exactly. Old joint choices are cached_verified and not
  separately reoptimized; new four-arm choices are recomputed.
- Main summary SHA256:
  `853779b5b791a742f2e7aa60924111210be519f691fd3acf2d23cd6ad044837e`.
- Complete receipt SHA256:
  `10680d6947c09a47f2a706c1ed2d7681502ff2acffa8cccbb254e1253c246479`.

## Engineering and report checks

The focused18-file suite passes65 tests in9.62s. It covers exhaustive small
allocation optima, input binding, canonical order, interrupted recovery,
feasible solver fallbacks, unknown-bound arithmetic, actual artifact integrity
and a selected-risk nonmonotonicity counterexample. This is not a full repository
test run. Earlier65-test execution had3 failures in the new report-only seed
reducer because its one-seed slice retained the three-seed reducer setting.
That setting was fixed; registered scientific code/results were not changed.

The898,645B aggregate archive verifies all72 exact JSON byte payloads. It
contains no row data, raw trajectories, checkpoints, image/video or third-party
dataset files. Uncompressed generated `groups/` is ignored locally; archive
verification does not remove it. No unrelated staged files are committed.

Repeated successful generation produces identical readout, diagnostics and
both SVG hashes. SVGs parse as XML and their private PNG previews are visually
checked for readable labels, intervals and lack of overlap. A first shell
checksum attempt failed because of an unsupported locale; it was not accepted
as validation. The successful comparison explicitly sets `LC_ALL=C`, checks
both command exit codes and compares nonempty SHA256 outputs.

## Scientific result is not an engineering pass count

Six of eight fitting checks pass. All-query optimality fails because4 physical
and7 neural solver outputs fail postcheck and retain feasible warm anchors.
Complete-recording risk fails: the neural joint policy supports47/444 views,
with322 known easy-risk violations. The special neural/joint interaction
contrast crosses zero. These negative results remain in every current report.

All data remain exposed SOURCE fitting; no independent or whole-pipeline OOF
confirmation. Zero new neural training in this experiment. No deployment,
Stage5C execution, SMC, metric/time-calibration or submission-readiness claim.
