# Evidence Revision, Boundaries, and Remaining Gaps

## Revision Scope

Fresh work: revised English manuscript, extraction/verification code, two vector
figures, Chinese operating tutorial and official venue check. Cached-verified
evidence: sixteen public aggregate analyses, four predictor/action code files,
and the local EqMotion producer manifest. No new model fitting, target readout,
bootstrap, independent calibration, final confirmation or deployment.

The old manuscript is unchanged. A suspected family-label issue was investigated,
not assumed: the early data utilities use Transformer ancestry, but
`run_m3w_cost_head_transfer.py` explicitly supplies native EqMotion predictions;
`run_m3w_eqmotion_cost_refit.py` installs excluded EqMotion cost producers.
The later cost experiments inherit this chain. The old EqMotion labels are
correct. The new local check validates twelve outer views and 36 inner groups
against the archived predictions and exclusions; it does not reload checkpoints.

## Claim-to-Evidence Map

| Claim | Evidence | Limit |
|---|---|---|
| Forest is better at same-count risk ranking in these views | `risk_ranking_v1`, contrast forest_strict_minus_neural_ratio | Development contrast, not original-primary superiority or architecture isolation |
| Matching the neural empirical loss did not fix ranking | `fraction_square_v1`, strict and ratio contrasts | Particular estimator/objective/budget; not all neural models |
| Selected harm remains underestimated | `fraction_square_v1/conditional_quality` | Mean continuous costs on selected complete outcomes, not probability calibration |
| Gain ranking violates observed easy ceiling | Eight-row table, every site/seed retained | Observed outcomes only; absent labels not certified |
| New joint opportunities are sparse | `protected_joint_support_v1` | Causal proxy audit, no accuracy readout |
| Existing calibration reuse is invalid | `calibration_support_v1` | Does not rule out a new appropriately excluded design |

The table/figure exporter verifies equal-site arithmetic, source hashes and
exact export reconstruction. It does not mechanically prove every prose claim.
Tables in the manuscript are regression-tested against generated values. The
new figures were also visually inspected for labels, clipping and sign conventions.

## Literature Verification

Reopened primary records on 22 September: EqMotion and JFP author abstracts/
metadata, Multi-Expert Deferral and Selective Regression proceedings, ExtraTrees
publisher metadata/abstract, Learn then Test v5 Sections 1.1 and 2.1, and the SDD
release page. These support the narrowly stated positioning, not an exhaustive
novelty assessment. No full-paper human-read attestation is inferred.

During drafting the Selective Regression author list was incorrect; it was
corrected from the proceedings to Abhin Shah and coauthors before verification.
The ExtraTrees volume is 63, checked at the publisher. Learn then Test's HTML
displays conflicting header dates, so the citation identifies arXiv version 5
without inventing a final venue/date. Dataset statistics here come from the
local audited population, not the release's total dataset count.

## Priority Gaps

1. **Independent data roles:** source-use/exposure and calibration/confirmation
   decisions remain pending. Four repeatedly explored sites cannot be renamed
   independent. No new source role is assigned by this draft.
2. **Method contribution:** original forest superiority and matched neural-loss
   superiority fail. The positive same-count contrast is a useful diagnostic,
   not a substitute primary result or a new method by itself.
3. **Conditional protection:** independent calibration, missing-outcome easy
   safety and a justified risk functional remain unresolved. A 2% ADE-degradation
   criterion is not a 2% probability-of-harm criterion.
4. **Joint contribution:** only sparse protected opportunities are demonstrated;
   useful predictive composition remains unproved. New labels or pair-weight
   searches are not opened by this audit.
5. **Release:** venue-specific layout, anonymous clean-environment reproduction,
   independent review and final author approval remain incomplete. Public repo
   history makes this package identifiable; it is not an anonymous supplement.

## Resource and Execution Record

Local files and GitHub main were verified at `412f96d7` before this revision.
Current SSH config still does not establish a CREATE alias/project path; no new
remote login, queue read or job submission was performed. This is not a finding
that remote assets are absent. Documentation/export is appropriate locally.

The first exporter attempt requested a CV p99 field absent from the aggregate
schema; it stopped before producing exports. That unsupported field was removed,
not invented or reconstructed from new labels. Existing model p95/p99 and CV p95
remain available. No source analysis was edited. Figure generation initially
built a local font cache; it finished and was not a training hang.

Seventeen scoped tests pass (six frozen-v1 tests and eleven new tests). Ten
generated exports, including both SVGs and the optional lineage receipt,
reconstruct exactly. The figures were inspected as local raster previews.
Verification receipts and test outcome are also recorded in the continuation
handoff and results ledger. Export checks are scoped, not the full
nonhermetic legacy test suite. No external scientific reviewer or paper-blind
review was used. All scientific completion gates remain unchanged.
