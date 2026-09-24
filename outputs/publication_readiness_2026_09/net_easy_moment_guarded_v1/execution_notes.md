# Execution and Verification Record

## Scope and Roles

Completed locally on 2026-09-24. This is fresh fitting of 36 risk forests and a
fresh fixed-policy readout, not new neural forecast training. Frozen forecasts,
source features, outcome labels and existing cost heads were hash-verified.
Four SDD physical sites remain design-exposed, even where excluded from a fit.
The protocol is obs8/pred12 native annotation steps, stride 12, annotation pixels.
Historical t+50 results are not pooled into these comparisons.

No independent calibration, confirmation, new external readout or deployment
was run. DroneCrowd remains reserved and closed. DUT remains exposed diagnostic;
HT21/CroHD remains source-audit/quarantine only. Stage5C and SMC remain off.

## Training

- Runtime: native arm64 `.venv-pytorch`, CPU threads 4, interop threads 1,
  DataLoader workers 0. No torch resource probing or multiprocessing.
- Retained environment: NumPy 2.4.6, PyTorch 2.12.0, scikit-learn 1.8.0,
  SciPy 1.17.1.
- Three actions x four excluded-site views x three seeds (17, 29, 43).
- All 36 heads reached the fixed 128-tree budget. The 16-tree pilot resumed;
  no discarded successful fit or outcome-based hyperparameter search.
- 768,000 draws per head, zero unknown-label training draws. The complete-label
  fitting estimand differs from available-point ADE evaluation support.
- Checkpoints every 16 trees, per-fit receipts, hashes and heartbeat retained
  locally. Checkpoints and large caches are not committed.
- Sum of fitting-loop times: 918.9900997910299 seconds. This excludes loading,
  allocation, evaluation and verification; it is not end-to-end wall time.
- Final per-head mean weighted training MSE ranges from 0.0294318685 to
  0.0456340521. These are fitting losses, not out-of-site prediction accuracy;
  the target set changed, so cross-experiment mean-loss rankings are invalid.

Full per-target traces and fitting receipt summaries appear in
[training_losses.md](training_losses.md) and [analysis.json](analysis.json).

## Version Boundary

The original training source and checkpoints remained immutable while review
identified decision/readout defects. The supported guarded runner fixes strict
selected-risk accounting, action/view/seed provenance and first-readout versus
replay checks. A second pre-readout review repaired JSON serialization and
verification of explicitly nonoptimal fallbacks. No decisions or outcomes had
been read under the superseded guarded identity.

Training uses `scripts/run_m3w_net_easy_moment.py`; decision/readout uses only
`scripts/run_m3w_net_easy_moment_guarded.py`. Original unguarded decision modes
are archival, not the supported route. The [review](code_review.md) is a bounded
code review, not an independent replication of this research.

Registration was locally hash-bound before fitting. The first public progress
commit, `a870b4f`, occurred during training and claimed no result. This is not an
externally preregistered confirmatory trial.

## Full Execution

| Step | Retained PID | Outcome |
|---|---:|---|
| Original preflight | 82498 | Completed |
| 16-tree pilot | 82676 | Completed and resumed |
| Full fixed training | 82911 | Exit 0; all 36 heads complete |
| Fresh guarded decisions | 84256 | Exit 0; full registered population |
| First aggregate readout | 84933 | Exit 0 |
| Exact decision replay | 85008 | Exit 0 |
| Exact aggregate replay | 86027 | Exit 0 |
| Separate arithmetic verification | Session 75080 | Exit 0 |
| Numerical sensitivity audit | Session 53894 | Exit 0; no decisions changed |
| Figure generation | Synchronous | Exit 0; rendered preview inspected |

The allocation event span was 477.899464 seconds, not whole-pipeline runtime.
All required sessions are terminal; no training remains hidden in the background.
Use the [Chinese recovery guide](operation_zh.md) for exact commands and paths.

## Verification

- All 175,756 indexed windows and 188,388 query/action/seed instances processed.
  Nine policy arms produce 14,236,236 decision bits, bound by 756 chunk receipts.
- Full decision and aggregate replays match the immutable original artifacts.
- Separate arithmetic passes 565,164 predicted-risk checks, 882 fixed small-query
  exhaustive optimum checks, 1,620 site/seed/subset reductions and 54 paired
  contrasts. Three large queries are explicitly skipped for exhaustive checking.
- This is a separate arithmetic implementation run by the same agent, not
  independent research confirmation.
- Three strict primal/dual checks fail closed. One Transformer matched-count
  query therefore fails its requested count. These are recorded, not silently
  called optimal or globally exact-count matched.
- All saved new budget-constrained decisions satisfy their predicted budgets.
  That statement is not observed safety or a calibration guarantee.
- The unmatched Transformer query can change primary gain by at most
  0.000946966730 percentage points. The analogous reported damping-positive and
  EqMotion-net bounds are 0.000171229942 and 0.000923989826 pp. These are diagnostic
  post-decision upper bounds, not outcome-selected replacement decisions.
- 144 scoped tests passed in 3.88 seconds. The unchanged tested core version is
  reused; the separate verifier's later monotonicity checks were exercised by
  the successful full arithmetic run. The unrelated legacy suite was not rerun.
- The SVG includes all 27 fixed controls and uncertainty, with no cherry-picked
  axis exclusion. Its private PNG preview was visually inspected.

## Immutable Identities

| Artifact | SHA-256 |
|---|---|
| Training identity | `113741aa574389bd0691a8ee2d2e0e0446213b5671eae8147d9573d8bc2c883f` |
| Guarded identity | `c007b712b233a237cb11b2d8b0cfb465749881d282b360fa958cfec86dc53267` |
| Decision manifest | `fbcaa5b3232ef552d53a09a2281afca752216050b53d88c4e2f67311141bcab8` |
| Analysis | `6780efc6c4b4a3759229ad157866a5fb1ed894ade1e0b30af99c7e9ebd004a31` |
| Separate verifier | `66f03ce4add9d9758eb6a552ac81a920a783adab6fa13541cf534c781d8edee4` |

## Claims Not Established

Neither reproduction nor a nominal four-site bootstrap is an independent
generalization result. The Transformer contrast over old strict spans zero;
EqMotion harms easy cases more than old strict. Zero-CV harm persists. Broader
coverage explains much of the unrestricted gain. No neural indispensability,
multimodal interaction contribution, certified safety, metric/seconds claim,
new deployment, true-3D model, foundation model or submission readiness follows.
