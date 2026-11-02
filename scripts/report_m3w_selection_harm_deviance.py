"""Complete controlled training/readout report; public scalar inputs only."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selection_harm_deviance as run


def load(ref):
    p = ROOT/ref['path']
    if not p.is_relative_to(ROOT/'outputs') or hashlib.sha256(p.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Verified public scalar receipt required')
    return json.loads(p.read_text())


def pool_fixed_costs(families):
    keys = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')
    if set(families) != set(run.api.ARMS) or any(not cells for cells in families.values()):
        raise ValueError('All three fixed-bank cost families required')
    pooled = {}
    for arm, cells in families.items():
        count = sum(c['known_action_views'] for c in cells)
        if count <= 0 or any(c['known_action_views'] < 0 for c in cells):
            raise ValueError('Observed action support required')
        mse = {k:sum(c['known_action_views']*c['normalized_MSE'][k]
                     for c in cells if c['known_action_views'])/count for k in keys}
        mass = {k:sum(c['observed_known_mass'][k] for c in cells) for k in keys}
        if any(not math.isfinite(v) or v < 0 for v in (*mse.values(), *mass.values())):
            raise ValueError('Finite nonnegative fixed-bank diagnostics required')
        pooled[arm] = dict(known_action_views=count, normalized_MSE=mse, observed_known_mass=mass)
    reference = pooled['mean_absolute']
    for cell in pooled.values():
        if cell['known_action_views'] != reference['known_action_views'] or any(
                not math.isclose(cell['observed_known_mass'][k], reference['observed_known_mass'][k],
                                 rel_tol=1e-10, abs_tol=1e-8) for k in keys):
            raise ValueError('Fixed observed bank support and target masses must match')
    return pooled


def main():
    public = run.PUBLIC; read = lambda n:json.loads((public/n).read_text())
    cfg = json.loads(run.CONFIG.read_text()); complete = read('complete.json'); verification = read('verification.json')
    rows = [load(r) for r in complete['groups']]; summary = read('summary.json')
    if (run.summarize(rows, cfg) != summary or not verification['all18_new_cost_inferences_exact']
            or not verification['all54_metric_groups_exact'] or not verification['mean_control_exact']
            or verification['summary_sha256'] != run.sha(public/'summary.json')):
        raise ValueError('Complete original-control and new-model replay required')
    freeze = read('training_freeze.json'); fits = [load(r) for r in freeze['fits']]
    if len(fits) != 18 or any(r['minimum_query_visits'] < 4 or r['unknown_supervised_rows'] for r in fits):
        raise ValueError('All18fits require complete known-supervision coverage')
    full = read('full_mean_control_refit.json'); recovery = read('resource_resume_verification.json')
    if not full['model_optimizer_sampler_trace_exact'] or not recovery['real_disk_interruption_2000_to8192_full_state_exact']:
        raise ValueError('Actual original-control and resource recovery verification required')
    source_complete = ROOT/'outputs/publication_readiness_2026_09/european_calibration_source_fit_v1/complete.json'
    source_refs = json.loads(source_complete.read_text())['cases']
    controls = [load(ref) for ref in source_refs if ref['path'].endswith('_mean_absolute.json')]
    if len(controls) != 9 or {c['trial'] for c in controls} != {f['trial'] for f in fits}:
        raise ValueError('All nine matching SOURCE controls required')
    families = {'mean_absolute':[g['diagnosis']['all_legal'] for c in controls for g in c['groups']]}
    families.update({a:[f['SOURCE_fit_diagnosis']['all_legal'] for f in fits if f['arm'] == a]
                     for a in run.api.NEW_ARMS})
    fixed_costs = pool_fixed_costs(families)
    (public/'fixed_bank_cost_diagnosis.json').write_text(json.dumps(dict(
        result_source='fresh_aggregation_of_cached_verified_public_SOURCE_fit_diagnostics',
        SOURCE_fitting_resubstitution_not_validation=True, selected_pools_used=False,
        source_complete_sha256=run.sha(source_complete), training_freeze_sha256=run.sha(public/'training_freeze.json'),
        fixed_bank_support_and_target_masses_match=True, arms=fixed_costs), indent=2, sort_keys=True)+'\n')
    fmt = lambda v:'N/A' if v is None else f'{v:.6g}'
    def interval(r):
        return 'N/A' if r['CI95'] is None else f'{r["point"]:+.6f} [{r["CI95"][0]:+.6f}, {r["CI95"][1]:+.6f}]'
    lines = ['# Matched H/EH Loss Training: Complete', '',
        'Fresh_run:18complete Torch SOURCE cost fits, full first-producer refits,',
        '18new frozen cost inferences and complete54group development readout/replay.',
        'Cached_verified: mean_absolute control, forecast bank and SOURCE-only data/',
        'normalization. No new forecaster, feature, goal, dataset or threshold search.',
        'Selection is already exposed development; calibration arrays, confirmation',
        'and DroneCrowd are not opened. Original failed experiments stay unchanged.', '',
        '## Fixed Factors And Real Training', '',
        'The easy-only arm replaces EH squared error with the existing stable cost',
        'deviance. The dual arm replaces H and EH. B/R/ER and absolute gain anchoring',
        'stay squared. Initialization,29,253parameters, SOURCE rows, sampler, draws,',
        'optimizer and complete coverage match the original mean_absolute control.',
        'All three50->100recovery checks and the original100step control are exact;',
        'the complete original control refit is also exact. Both new first-producer',
        'full refits match. These are reproducibility checks, not generalization.', '',
        '|Arm|Fits|Primary updates|Mean first minibatch total|Mean last minibatch total|Mean last common MSE/gain objective|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.api.NEW_ARMS:
        cells = [r for r in fits if r['arm'] == arm]
        mean = lambda fn:sum(fn(r) for r in cells)/len(cells)
        lines.append(f'|{arm}|{len(cells)}|{sum(r["step"] for r in cells)}|'
            f'{mean(lambda r:r["trace"][0]["total"]):.6f}|{mean(lambda r:r["trace"][-1]["total"]):.6f}|'
            f'{mean(lambda r:r["trace"][-1]["original_total"]):.6f}|')
    lines += ['', 'Loss definitions differ; raw totals across arms are not a common test',
        'metric. First/last minibatches differ. The original-total monitor is the',
        'same fitting loss, not validation, and does not establish downstream lift.', '',
        '## Complete Development Policies', '',
        '|Policy|Known ADE gain % [95% CI]|Supported views|Known all/easy violations|Undefined easy|Worst signed easy net %|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        s = summary['safety'][arm]
        lines.append(f'|{arm}|{interval(summary["accuracy"][arm])}|{s["supported_record_views"]}/{s["required_record_views"]}|'
            f'{s["known_all_risk_violations"]}/{s["known_easy_risk_violations"]}|{s["undefined_easy_views"]}|'
            f'{fmt(s["worst_known_easy_net_degradation_percent"])}|')
    lines += ['', 'All38,102targets/28recordings/six localities and nine pipelines remain.',
        'The252risk views repeat recordings and are not independent samples. Unknown',
        'and partial outcomes are retained. Original2%all/easy positive-harm and',
        'easy-net checks apply to every required view; undefined support is not safe0.', '',
        '## Registered Contrasts', '',
        '|Arm minus mean_absolute|Known pp [95% CI]|Direct paired completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for arm in run.api.NEW_ARMS:
        e = summary['effects'][arm]
        lines.append(f'|{arm}|{interval(e["known_pp"])}|{interval(e["completion_lower_pp"])}|')
    lines += ['', 'Registered primary is the dual arm FULL paired completion-lower effect.',
        f'Primary pass: **{summary["primary_pass"]}**. Original record-risk pass:',
        f'**{summary["original_record_risk_pass"]}**. Candidate: **{summary["candidate_pass"]}**.',
        'No alternate arm or favorable mean replaces a failed registered primary.',
        'Intervals use3,000locality resamples, averaging repeated folds/actual training',
        'seeds17/29/43 within each site. These are conditional development intervals.',
        'Unknown-pair disagreement is computed directly, not as two lower bounds',
        'subtracted. No population-risk guarantee or independent confirmation follows.', '',
        '## SOURCE Cost Diagnosis', '',
        '|Arm|Pooled selected H predicted/true|Pooled selected EH predicted/true|Known selected all/easy risks %|',
        '|---|---:|---:|---:|']
    for arm in run.api.NEW_ARMS:
        cells = [r['SOURCE_fit_diagnosis']['selected'] for r in fits if r['arm'] == arm]
        p = {k:sum(c['predicted_known_mass'][k] for c in cells) for k in ('harm','easy_harm','reference','easy_reference')}
        y = {k:sum(c['observed_known_mass'][k] for c in cells) for k in p}
        ratio = lambda a,b: None if b <= 0 else a/b
        lines.append(f'|{arm}|{fmt(ratio(p["harm"],y["harm"]))}|{fmt(ratio(p["easy_harm"],y["easy_harm"]))}|'
            f'{fmt(ratio(100*y["harm"],y["reference"]))}/{fmt(ratio(100*y["easy_harm"],y["easy_reference"]))}|')
    lines += ['', 'These pool repeated SOURCE fitting action views, not held-out validation.',
        'Selected pools differ by arm; their prediction ratios are not fixed-set error reductions.',
        'Improved fit or cost ratios cannot substitute for the development policy and',
        'original risk checks. The old144-fit easy-deviance failure is distinct and',
        'remains negative; this adaptation is not a new divergence contribution.', '',
        '### Fixed All-Legal Bank: Common Normalized Squared Error', '',
        '|Arm|Observed action views|H MSE|EH MSE|R MSE|ER MSE|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        c = fixed_costs[arm]; m = c['normalized_MSE']
        lines.append(f'|{arm}|{c["known_action_views"]}|{m["harm"]:.8g}|{m["easy_harm"]:.8g}|'
                     f'{m["reference"]:.8g}|{m["easy_reference"]:.8g}|')
    lines += ['', 'Counts and observed target masses match across arms; normalization and bank stay fixed.',
        'These repeated fitting actions are not independent held-out data. Dual H MSE decreases,',
        'but EH MSE increases; a better selected-pool ratio is not blanket cost accuracy.', '',
        '## Runtime And Actual Resource Recovery', '', '|Phase|PID|Seconds|Peak RSS bytes|', '|---|---:|---:|---:|']
    for name in ('pilot','training_freeze','policy_freeze','complete','verification','resource_resume_verification'):
        r = read(name+'.json'); lines.append(f'|{name}|{r["pid"]}|{r["seconds"]:.6f}|{r["peak_RSS_bytes"]}|')
    lines += ['', 'The first training process hit the10GiB disk-reserve guard after six fits',
        'and checkpoint2000/8192of single1_seed17/easy_deviance. Free space recovered',
        'without deleting any data/cache/system file; the fluctuation cause is not',
        'established. The unchanged run resumed. Training-freeze runtime covers the',
        'completion process only, not the earlier interrupted process. Per-fit traces',
        'and optimizer time are retained. A separate direct8192update refit matches',
        'the actually resumed full model/optimizer/sampler/draws/trace/RNG exactly.',
        'No budget/guard reduction. CPU4/inter-op1/workers0, native arm64.',
        'CREATE queue access was verified read-only; no remote jobs submitted/changed.', '',
        '## Reproduction And Boundaries', '', '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py pilot',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py infer --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py eval --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py verify',
        '.venv-pytorch/bin/python scripts/verify_m3w_harm_training_resource_resume.py',
        '.venv-pytorch/bin/python scripts/report_m3w_selection_harm_deviance.py', '```', '',
        'Completed phases are immutable; do not delete them to rerun. Unchanged partial',
        'training resumes from its full atomic checkpoints. Private SOURCE files and',
        'weights are required for real execution; this is not a cold-start release.',
        'No learned deployment changes. Available-label ADE is not full12 ADE; bounds',
        'cover wholly unknown rows, not missing steps inside partial futures. Raw',
        'stride12/local pixels/silver are not seconds/meters/human gold. No true3D,',
        'foundation or new trajectory-dynamics claim. Stage5C/SMC remain off.', '']
    (public/'readout.md').write_text('\n'.join(lines))
    (public/'gates.md').write_text('# Controlled Training Gates\n\n'+
        '\n'.join(f'- {k}: {v}' for k,v in dict(complete18fits=True, exact_control_and_recovery=True,
        full_source_coverage=True, full18inference54metric_replay=True, primary=summary['primary_pass'],
        original_record_risk=summary['original_record_risk_pass'], independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c=False, smc=False).items())+'\n')
    print(json.dumps(dict(complete_fits=18, primary=summary['primary_pass'], original_risk=summary['original_record_risk_pass'])))


if __name__ == '__main__': main()
