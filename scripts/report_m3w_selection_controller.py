"""Reproduce the controller report from verified public scalar groups, not private data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m3w_selection_controller import summarize

HOME = ROOT/"outputs/publication_readiness_2026_09/european_selection_controller_v1"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verified():
    cfg = json.loads((ROOT/"configs/m3w_selection_controller_v1.json").read_text())
    complete = json.loads((HOME/"complete.json").read_text())
    verification = json.loads((HOME/"verification.json").read_text())
    summary = json.loads((HOME/"summary.json").read_text())
    if (verification["summary_sha256"] != sha(HOME/"summary.json") or complete["summary_sha256"] != sha(HOME/"summary.json")
            or verification["complete_sha256"] != sha(HOME/"complete.json")
            or not verification["all18_cost_inference_exact"] or not verification["all54_frozen_action_metric_groups_exact"]):
        raise ValueError("Complete inference/frozen-action replay required")
    rows = []
    for ref in complete["groups"]:
        path = (ROOT/ref["path"]).resolve()
        if path.parent != HOME/"groups" or sha(path) != ref["sha256"]:
            raise ValueError("Only hash-bound public scalar groups admitted")
        rows.append(json.loads(path.read_text()))
    if summarize(rows,cfg) != summary: raise ValueError("Public scalar summary replay differs")
    fitfreeze = json.loads((HOME/"training_freeze.json").read_text()); fits = []
    for ref in fitfreeze["fits"]:
        path = ROOT/ref["path"]
        if path.parent != HOME/"fits" or sha(path) != ref["sha256"]: raise ValueError("Cost fit receipt changed")
        fits.append(json.loads(path.read_text()))
    return summary, rows, fits, fitfreeze, complete, verification


def render():
    s, rows, fits, training, complete, verify = verified()
    ci = lambda x:f"{x['point']:+.4f} [{x['CI95'][0]:+.4f}, {x['CI95'][1]:+.4f}]"
    values = []
    for name, accuracy in s["accuracy"].items():
        risk = s["safety"][name]
        intervention = sum(r["policies"][name]["switch_rate"] for r in rows)/len(rows)*100
        values.append(f"|{name}|{ci(accuracy)}|{intervention:.2f}%|"
            f"{risk['worst_known_easy_net_degradation_percent']:.2f}%|{risk['supported_record_views']}/{risk['required_record_views']}|"
            f"{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']}|{risk['undefined_easy_views']}|")
    by_site = [f"|{k}|{s['primary']['known_pp']['by_locality'][k]:+.2f}|{s['primary']['completion_lower_pp']['by_locality'][k]:+.2f}|"
               for k in sorted(s["primary"]["known_pp"]["by_locality"])]
    nonempty = sum(v["nonempty"] for r in rows for v in r["count_matched_record_support"].values())
    seed_rows = []
    for seed in (17,29,43):
        cells = [r for r in rows if r["seed"]==seed]
        known = sum(r["primary_joint_minus_independent"]["known_pp"] for r in cells)/len(cells)
        lower = sum(r["primary_joint_minus_independent"]["completion_lower_pp"] for r in cells)/len(cells)
        seed_rows.append(f"|{seed}|{known:+.2f}|{lower:+.2f}|")
    loss_rows = []
    for arm in ("row","context"):
        cells = [r for r in fits if r["arm"]==arm]
        loss_rows.append(f"|{arm}|{len(cells)}|{sum(r['budget'] for r in cells)}|"
            f"{sum(r['first_trace']['total'] for r in cells)/len(cells):.5f}|"
            f"{sum(r['final_trace']['total'] for r in cells)/len(cells):.5f}|")
    primary = s["primary_pass"]; risk = s["original_record_risk_pass"]
    content = f"""# SOURCE-To-Selection Cost Controllers: Complete Readout

## Result Source And Scope

Fresh_run: nine SOURCE forecast caches,18real Torch cost fits, two first-producer
complete refits, new selection cost inference, causal allocation and54metric
groups. Cached_verified: actual source-trained forecasting checkpoints, original
SOURCE whitelist, frozen selection inputs/forecasts and previously verified
model-selection labels. No new trajectory model was trained. The six selection
localities were already exposed by the supporting forecast readout, not an
untouched final test. Risk calibration, confirmation and DroneCrowd stay closed.

Every cost model fits only the eight SOURCE localities outside its forecaster's
four training localities. Fold costs use91,562/145,422/62,036rows from the original
149,510-row whitelist, repeated across actual forecasting seeds17/29/43.
The same rows are not independent new examples. No selection labels train the
weights, normalizers or thresholds. All38,102selection targets/28recordings/six
localities remain, including futureless and partially labelled targets.

## Controlled Method

Both29,253-parameter critics share initialization, complete query draws and
physical-only SOURCE feature normalization. Inputs are380causal geometry/rollout
and64past detector fields. Only the parameter-free mean of other same-query
target representations differs. Original partial-neighbor context remains in
both arms. No scene ID, raw future endpoint, central velocity, future support,
test statistic or test goal enters inference. Cost is benefit/positive-harm/
reference/easy-reference/easy-harm, not hard best-baseline classification.

The decoder shares references across actions and preserves B<=R,ER<=R,EH<=H and
B+H<=causal disagreement. Known-only underharm/ranking objectives reuse existing
cost learning. Grouped risk pooling reuses the existing allocator; this is not
a newly invented deferral loss, new JEPA/dynamics, pairwise interaction
optimization or a claim of physical safety. The physical-bank comparison
removes the neural action from the same critic, not a retrained ablation.

## Registered Primary

Context matched-joint minus context independent, relative to the same fixed
SOURCE-selected floor. Intervention counts match within every query.

|Contrast|Effect, percentage points [95% locality bootstrap CI]|
|---|---:|
|Registered paired completion-lower primary|{ci(s['primary']['completion_lower_pp'])}|
|Known-label secondary|{ci(s['primary']['known_pp'])}|
|Paired completion-upper diagnostic|{ci(s['primary']['completion_upper_pp'])}|
|Context minus row matched-joint, known secondary|{ci(s['context_effect']['known_pp'])}|
|Context minus row matched-joint, completion lower|{ci(s['context_effect']['completion_lower_pp'])}|
|Neural-action bank minus physical bank, known secondary|{ci(s['neural_action_effect']['known_pp'])}|
|Neural-action bank minus physical bank, completion lower|{ci(s['neural_action_effect']['completion_lower_pp'])}|

Primary pass: **{primary}**. Original complete-record risk pass: **{risk}**.
Candidate gate: **{s['policy_candidate_pass']}**. Deployment changed: **false**.
No favorable secondary, free-count result or seed replaces the registered primary.

## Complete Accuracy, Intervention And Risk

|Policy|Known available-label ADE gain vs floor,% [CI]|Mean switch rate|Worst known easy net degradation|Supported repeated record views|Known all/easy risk violations|Undefined selected-easy views|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(values)}

The floor is each producer's SOURCE-selected fixed reference, not an oracle or
a baseline picked on these scenes. Raw_history and floor controls exactly match
the earlier complete selection readout. Rates average locality/fold/seed views,
not row-weighted rates. There are252repeated recording-risk views, not252unique
recordings. Positive harm cannot be canceled by benefit in the risk numerator.
The original all/easy positive-harm and easy-net budgets stay2% on every record.
The floor adds no harm; an empty selected-easy reference is not estimable, not
proof that abstention is harmful or a successful positive learned intervention.
Nonempty count-matched utility controls: {nonempty}/252views. Empty comparisons
are labelled not_estimable and do not certify safety. Full-policy floor-versus-
floor utility can be exactly zero without proving any selection mechanism.

## Locality And Seed Effects

|Selection locality|Primary known component,pp|Primary paired lower,pp|
|---|---:|---:|
{chr(10).join(by_site)}

|Actual forecasting/training pipeline seed|Mean primary known component,pp|Mean primary paired lower,pp|
|---|---:|---:|
{chr(10).join(seed_rows)}

Intervals use3000locality resamples after averaging source folds/seeds within
each of six localities. Repeated forecasts/windows are not independent scenes.
These are conditional model-selection development intervals, not calibrated
population guarantees or independent confirmation. Unknown disagreement bounds
follow the previous available-label ADE convention and do not recover missing
steps of partial trajectories. Full-roster FDE is not estimable; available-endpoint
FDE in scalar groups is diagnostic. No future-completeness filter changes the roster.

## Real Training And Recovery

|Arm|Complete fits|Primary updates|Mean first minibatch loss|Mean final minibatch loss|
|---|---:|---:|---:|---:|
{chr(10).join(loss_rows)}

Every original fitting query has at least four visits. No unknown future has a
supervised-zero label. These are varying SOURCE minibatches, not validation losses
or generalization evidence. Complete scalar traces are exported separately.

The original400-update real pilot failed bit-exact context recovery by2.42e-08;
its log/checkpoints and original registration are preserved. Registration_v2
replaces scatter/gather reductions with fixed membership products. The repaired
real pilot and two full first-producer refits check exactness without a relaxed
tolerance. This verifies the repair on tested runs, not a unique hardware cause
or an OpenMP/SHM hang. No new policy outcomes were scored before that repair.

Formal training PID{training['pid']}: {training['seconds']:.3f}s, peak RSS
{training['peak_RSS_bytes']:,}B. Primary updates{training['primary_optimizer_updates']:,};
full-refit updates{training['exact_full_refit_updates']:,}. Readout PID{complete['pid']}:
{complete['seconds']:.3f}s. Separate verification PID{verify['pid']}: {verify['seconds']:.3f}s.
All18fresh cost inferences and54frozen-action metric groups replay exactly;
query feasibility and matched counts are checked. Solver optimization is **not**
claimed reexecuted by this verification. Local arm64CPU4/workers0 was adequate;
no CREATE job was submitted and no remote scheduler state is inferred.

## Remaining Paper Requirements

Do not equate a completed program with the paper hypothesis. A useful and safe
joint policy, conflict/composition controls, independent risk calibration/final
confirmation, appropriate public predictors and anonymous end-to-end reproduction
remain necessary. Selection results can choose the next method, not serve as
an untouched final test. No deployment or submission-readiness promotion here.

obs8/pred12/rawstride12, local-image pixel, detector-silver,2.5D/pseudo-3D agent
trajectory state modeling. Not metric, seconds-level, human gold, true3D or
foundation evidence. Stage5C is not executed and SMC is not enabled.
"""
    (HOME/"readout.md").write_text(content)
    report = dict(result_source="cached_verified_public_scalar_rebuild_of_fresh_training_and_eval",
        summary_sha256=sha(HOME/"summary.json"),readout_sha256=sha(HOME/"readout.md"),
        public_scalar_groups_exact=True,private_input_reads=0,primary_pass=primary,original_record_risk_pass=risk,
        submission_ready=False,deployment_changed=False)
    (HOME/"report_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


if __name__ == "__main__": print(json.dumps(render(),indent=2,sort_keys=True))
