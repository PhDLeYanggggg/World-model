"""Account for fixed-anchor abstention without using new repair outcomes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_risk_priority_policy as run
from scripts.report_m3w_easy_risk_priority import structural_support, put


def main():
    assert run.identity() == json.loads((run.PUBLIC/'decision_registration.json').read_text())
    source = run.original.PUBLIC
    seal = json.loads((source/'verification.json').read_text())
    replay_path = source/'evaluation_replay.json'
    assert run.digest(replay_path) == seal['artifacts'][replay_path.name]
    replay = json.loads(replay_path.read_text()); assert replay['exact']
    ref = replay['details']; assert run.base.artifact(ROOT/ref['path']) == ref
    rows = json.loads((ROOT/ref['path']).read_text())['rows']
    support = structural_support(rows)
    assert support['total_dependent_views'] == 216
    evidence = dict(result_source='fresh_run_support_reduction_of_cached_verified_controls',
        source_evaluation_replay=run.base.artifact(replay_path), source_details=ref,
        registered_decisions=run.base.artifact(run.PUBLIC/'decision_registration.json'),
        support=support, new_repair_decisions_used=False, new_repair_outcomes_used=False,
        threshold_changed=False, primary_changed=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable(run.PUBLIC/'structural_support.json', evidence)
    text = f'''# Fixed-Anchor Abstention: Pre-Readout Structural Limit

Source: fresh_run support reduction of cached_verified, sealed previous control
decisions. No new repair predictions or outcomes are used; no new scene is opened.

Among{support['total_dependent_views']} dependent source-role/seed/locality views,
the unchanged raw anchor has{support['raw_empty_views']} completely abstaining
views; the reproduced uncapped anchor has{support['uncapped_empty_views']}.
Their overlap is{support['overlapping_empty_views']}; the union is
**{support['forced_undefined_selected_risk_views']} views** across
{len(support['affected_localities'])} localities. These are not independent samples.

The registered common anchor is raw AND uncapped AND risk_priority. It is a
subset of each fixed anchor. If either fixed anchor selects no query member in
an entire view, common-count matching also selects none, regardless of the new
head. The selected-error denominator is then zero and selected risk is undefined.
No unobserved future label is needed for this implication.

Consequently the unchanged every-view-defined-selected-risk gate cannot pass in
this experiment. This is not evidence that the gradient repair worsens prediction;
its registered paired ADE comparison still has a meaningful, falsifiable role.
Full training and readout continue unchanged. Do not drop these views, count
undefined risk as zero, loosen thresholds or replace the original risk primary.

The original selected-risk research objective still needs separately justified
support and independent calibration/confirmation evidence. A favorable repair
readout alone would not supply that evidence or authorize deployment.
Image-local detector silver, raw-frame obs8/pred12 only. No metric/seconds,
physical-safety, true3D, foundation or submission claim. Stage5C/SMC disabled.
'''
    put('structural_support.md', text)
    print(json.dumps(support, indent=2))


if __name__ == '__main__':
    main()
