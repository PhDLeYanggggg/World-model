# Pre-Full-Run Reporting Repair

The first complete-head pilot (33.29 s) at registration commit `c7759278` is
preserved as `pilot.json`. No new full-cohort outcomes were evaluated before this
amendment. Code inspection identified that using ER > 0 to mark known easy
outcomes omits the legitimate case ER = 0 and EH > 0. A new regression test
reproduces the error. This is not a reason to drop zero-reference cases.

Repair the descriptive mask to ER > 0 or EH > 0 and independently assert that
reported selected known EH equals the original five-moment label sum. If both
ER and EH are zero, the omitted step-easy harm is also zero: either the row is
not easy or its zero reference and zero harm imply zero observed step errors.
Undefined zero denominators remain undefined, never a safe zero ratio.

Original policy, forecasts, fits, splits, masks, primary metrics, risk budget,
probe targets, comparison arms and the preregistered screen are unchanged. No
threshold or outcome-driven choice was added. Preserve `registration.json`;
`registration_amended.json` binds the repaired sources and both test files.
Run a fresh `pilot_amended.json` before all 72 heads. Report this repair in final
execution history. This is an accounting fix, not a model improvement.
