# Diagnostic Intake Admission: Quality Is Not Protocol Approval

Date: 2026-09-16. This is a data-admission repair with synthetic integration and real metadata refusal checks. It does not add real forecasting accuracy, independent scenes, source permission or a deployment claim.

## Reproduced Defect

The previous DUT audit found duplicated agent annotations and marked the recording for quarantine. That finding existed in the report, but `ExperimentContract` checked only scientific approval, roles, cache identity and declared historical use. A synthetic protocol marked approved could therefore admit a cache whose metadata still said `source_conditions_review = pending_not_formal_use_approval`, without resolving source conditions or annotation quality.

The new regression failed on the old implementation with `DID NOT RAISE ContractError`. This was a potential future admission defect, **not evidence that a real unauthorized experiment occurred**: the real protocol has remained unapproved throughout, and no real source was assigned a role to reproduce it.

## Repair

Active records whose cache carries `source_conditions_review`, or whose protocol explicitly supplies `intake_screen`, now require a hash-bound intake screen before a recording reader is opened. The screen is separate from the scientific approval and records:

- exact recording/cache metadata identity, physical site and source-file hashes;
- the verified author-source manifest, conversion report and independent quality audit;
- each recording's quality disposition and blocking findings;
- the audit's full recording coverage and row-count agreement.

The checker follows the underlying audit, not just a summary flag. For example, changing a DUT screen's disposition from `quarantine` to `no_flags_in_limited_screen` cannot clear a duplicate-agent finding still present in the bound audit. Known quarantine blocks fit, development, calibration and confirmation alike. The repair does not merge agents, rewrite raw data, remove windows or select a new split.

A non-quarantined screen is **not** source-use permission. The scientific protocol must separately bind `source_use_decision`: named reviewer, decision reference, allowed data roles and hash-bound decision evidence. An approval for fit does not cover confirmation. These declarations are included in the protocol digest; changing them after approval invalidates the protocol. The code verifies consistency and evidence identity, not a reviewer's identity, legal entitlement or the truth of the declaration. Historical exposure and physical-scene role checks remain separate and still apply.

Evidence is checked at contract construction and rechecked when the contract is used to open a recording, verify a prediction operation or start/resume fitting. A changed screen, source manifest, audit, conversion record or source-use evidence is refused. This is **not** continuous filesystem monitoring during every tensor operation; keep an active run's evidence immutable. Excluded records can remain in the inventory but cannot be opened for a usable role.

Existing canonical records without an intake-review marker retain the earlier declaration-based contract. That compatibility is not retroactive proof that their source conditions are resolved. Legacy scripts bypassing `ExperimentContract`, direct diagnostic readers and arbitrary external code are not converted into a system-wide access-control boundary. Future source additions must preserve their pending-review marker and register their evidence honestly.

Newly generated protocol drafts bind the intake-checker source code as well as the existing contract. **The current real draft was not modified or approved.** Its source bindings must be reviewed with the final implementation when the scientific decisions are resolved; do not edit hashes just to make an old approval pass.

## Real DUT Check

The [screen](dut_screen.json) binds the previously verified 28-clip DUT conversion and its [independent audit](../dut_causal_intake/independent_recount.json). Array/cache identities are verified without requesting future labels or predictions. The [refusal report](dut_refusals.json) makes 112 metadata-only checks: 28 recordings times four hypothetical roles, **not 112 actual role assignments**.

| Refusal | Checks | Meaning |
|---|---:|---|
| Unresolved quality quarantine | 4 | `dut_intersection_04`, one refusal per role |
| Separate source-use decision absent | 108 | Other 27 clips, one refusal per role |
| Admitted recordings | 0 | No scientific role or source approval issued |

The duplicated raw pedestrian IDs 10/11 still have identical coordinates at frames 1-145. The original 457,686 rows are preserved; no raw/cache data changed. The 28 clips still represent two source-described physical locations, not 28 independent confirmation scenes. Dataset-local/raw-frame boundaries, unknown historical predictive exposure and source-use review remain unresolved.

The source and conversion are `cached_verified`. The new screen binding and refusal checks are `fresh_run`. Real fitting, development selection, calibration and confirmation remain `not_run`.

## Positive and Negative Path Verification

New tests exercise missing review, all four role refusals, quarantine despite source approval, summary-flag concealment, source/audit/cache mismatches, malformed evidence, role-restricted decisions, path escape, approval digest drift, excluded data and evidence changes after construction.

A subprocess runs the real forecasting training CLI with a **synthetic quarantined** protocol. It exits 2 before Torch training or checkpoint-directory creation. A separate **admitted synthetic** fixture executes four real Torch CPU optimizer updates as 2 + resume + 2, verifies finite losses and changed parameters, produces four heartbeat records, and returns `cached_verified` on completed resume. Changing source-use evidence then prevents another resume. This establishes that the admission gate does not simply disable all fitting; it is not real-data model improvement or a new MPS stability test.

An initial metadata-only refusal run exposed an error when a descriptive pending string reached the decision-object parser. Type validation now produces a structured refusal, a regression covers that case, and the final real check was rerun. Only this turn's incomplete generated screen was rebuilt; prior source/audit/conversion evidence was preserved.

Final test counts, exact code/report hashes and preflight outputs are recorded in [verification.json](verification.json). The historical nonhermetic full-suite result (1,870 pass / one unrelated data-lake fixture failure) is not rewritten by this focused regression run.

## Reproduction and Remaining Work

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py -q
.venv-pytorch/bin/python scripts/build_m3w_dut_admission_screen.py --output outputs/publication_readiness_2026_09/intake_admission/new_screen.json --report outputs/publication_readiness_2026_09/intake_admission/new_refusals.json
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --preflight-only
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --preflight-only
```

The builder refuses to overwrite an existing screen/report. Use new output names to preserve past evidence. The last two commands should still refuse the real unapproved protocol with exit 2. They do not silently approve roles or start training.

The next scientific step still depends on the previously requested protocol/independence decisions and source eligibility. A marked quarantine needs annotation review and a new justified audit or explicit exclusion, not a threshold adjustment. No new CREATE connection or job was attempted because the access conditions have not changed. Stage5C/SMC remain disabled, no metric/seconds/foundation claim is added, and CVPR submission readiness remains unproven.
