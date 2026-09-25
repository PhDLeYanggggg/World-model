"""Aggregate every registered nested-calibration result without selecting a winner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.report_m3w_european_cv_reference import sha, value, ci
from scripts.report_m3w_european_protected_motion import compact_description, compact_metric

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'


def accounting(a):
    expected = sha(PUBLIC/'analysis.json')
    for name in ('verification.json', 'checkpoint_replay.json'):
        r = json.loads((PUBLIC/name).read_text())
        if r['analysis_sha256'] != expected or not r['all_passed']:
            raise ValueError('Complete metric and checkpoint replay required')
    checks = json.loads((PUBLIC/'checkpoint_replay.json').read_text())['checks']
    if len(checks) != 54 or any(not r['exact'] or r['unknown_draws'] or r['supervised_draws'] != 512000 for r in checks):
        raise ValueError('All score-head replays and known-label draws required')
    pid = a['identity']['producer_identity']
    packed = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
    r = json.loads((packed/'receipt.json').read_text())
    if r['identity'] != pid['parent_identity']:
        raise ValueError('Wrong source population')
    for name in ('history', 'sites'):
        if sha(packed/(name+'.npy')) != r['arrays'][name]:
            raise ValueError('Changed accounting inputs')
    h = np.load(packed/'history.npy', mmap_mode='r', allow_pickle=False)
    sites = np.load(packed/'sites.npy', mmap_mode='r', allow_pickle=False)
    audited = []
    for receipt in a['decisions']:
        path = ROOT/receipt['path']
        if sha(path) != receipt['sha256']:
            raise ValueError('Decision receipt changed')
        d = json.loads(path.read_text())
        if d['uses_future_input'] or d['identity'] != a['identity'] or sha(ROOT/d['path']) != d['sha256']:
            raise ValueError('Invalid outer inference record')
        cal = d['calibration']
        if sha(ROOT/cal['path']) != cal['sha256']:
            raise ValueError('Fitted calibration map changed')
        c = json.loads((ROOT/cal['path']).read_text())
        roles = c['roles']
        if any(set(roles[x]) & set(roles[y]) for x,y in [('fitting','calibration'),('fitting','readout'),('calibration','readout')]):
            raise ValueError('Calibration role overlap')
        with np.load(ROOT/d['path'], allow_pickle=False) as z:
            ids, stored = z['ids'].copy(), z['switch'].copy()
        if set(sites[ids]) != set(roles['readout']):
            raise ValueError('Wrong outer readout population')
        result = {}
        for task in ('utility', 'risk'):
            art = c[task]['scores']
            if sha(ROOT/art['path']) != art['sha256']:
                raise ValueError('Frozen score bank changed')
            with np.load(ROOT/art['path'], allow_pickle=False) as z:
                pos = np.searchsorted(z['ids'], ids)
                np.testing.assert_array_equal(ids, z['ids'][pos])
                result[task] = z['scores'][pos].copy()
        rule_name = Path(d['path']).stem.split('_cal', 1)[1].split('_', 1)[1]
        config = c['rule']['rules'][rule_name]
        u, m = result['utility'], result['risk'].copy()
        moving = np.linalg.norm(np.diff(h[ids], axis=1), axis=2).sum(1) > 0
        threshold = .02
        if rule_name == 'population_rescale':
            m *= [config['denominator_multiplier'], config['harm_multiplier']]
        elif rule_name == 'selected_risk_grid' and not config['abstain']:
            threshold = config['threshold']
        expected_bits = moving & (u[:,0] > u[:,1]) & (m[:,0] > 0) & (m[:,1] <= threshold*m[:,0])
        if config['abstain']:
            expected_bits[:] = False
        np.testing.assert_array_equal(expected_bits, stored)
        audited.append(dict(receipt=receipt['path'], rows=len(ids), exact=True, role_sets_disjoint=True))
    if len(audited) != 216 or len(a['policies']) != 72:
        raise ValueError('Incomplete fixed calibration matrix')
    return dict(analysis_sha256=expected, result_source='fresh_accounting_cached_verified_models',
        decision_checks=audited, all_passed=True, scorer_replays=54, inner_producer_replays=18)


def summarize(a):
    return dict(result_source=a['result_source'], source_rows=a['source_rows'],
        policies={k: compact_description(v, scenes=True) for k,v in a['policies'].items()},
        neural_vs_damping={k: {n: compact_metric(m) for n,m in v.items()} for k,v in a['neural_vs_damping'].items()},
        calibration_vs_none={k: {n: compact_metric(m) for n,m in v.items()} for k,v in a['calibration_vs_none'].items()},
        new_inner_predictors=18, new_forecast_updates=72000, new_heads=54, new_head_updates=108000,
        orientations=2, seeds=[17,29,43], bootstrap_resamples=3000, bootstrap_unit='source_locality',
        uncertainty='conditional_source_development_not_independent_confirmation',
        independent_reserved_readout=False, deployment_changed=False, calibrated_safety=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    audit = accounting(a)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    summary = summarize(a)
    summary['analysis_sha256'] = audit['analysis_sha256']
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n')
    lines = ['# Nested Calibration Results', '',
        'All 72 pointwise views; no outer-result model/threshold/seed selection. Three seeds and both predefined role rotations.',
        'Forecasts and all scoring-head preprocessing exclude the internal calibration and outer localities.',
        'Four fitting, four calibration and four outer localities per assignment; all are opened development sources.', '',
        '| Seed/rotation/candidate/event/rule | ADE vs CV (%) | Conditional 95% CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for key,p in a['policies'].items():
        easy = p['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']
        e = 'undefined' if easy is None else f'{-easy:.6f}'
        lines.append(f"| {key} | {value(p['ADE_vs_CV'])} | {ci(p['ADE_vs_CV'])} | {value(p['FDE_vs_CV'])} | {value(p['hard_ADE_vs_CV'])} | {e} | {p['zero_CV']['harmed_rows']}/{p['zero_CV']['rows']} | {100*p['switch_rate']:.6f} |")
    for title, field in [('Neural Versus Matched Damping', 'neural_vs_damping'), ('Calibration Versus Its Uncalibrated Control', 'calibration_vs_none')]:
        lines += ['', '## '+title, '', '| View | Subset | Paired gain (%) | Conditional CI |', '|---|---|---:|---|']
        for key,row in a[field].items():
            for subset,m in row.items():
                lines.append(f'| {key} | {subset} | {value(m)} | {ci(m)} |')
    lines += ['', 'No joint optimization or independent risk certificate is claimed. A zero-switch result is fallback, not positive transfer.',
        'Bootstrap resamples localities 3,000 times conditional on shared source data/models. The two rotations are not independent replications.',
        'Unknown, negative and undefined results remain visible. Pixel obs8/pred12 rawstride12; not t50, seconds, metric, physical safety or human gold.',
        'No Stage5C, SMC, true3D, foundation, deployment promotion or submission-readiness claim.', '']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    cal_summary = []
    for item in a['calibration']['receipts']:
        c = json.loads((ROOT/item['path']).read_text())
        cal_summary.append(dict(name=Path(item['path']).stem, roles=c['roles'], rules=c['rule']['rules'],
            raw_evidence=c['rule']['raw_evidence'], grid_evidence=c['rule']['grid_evidence'],
            calibrated_guarantee=False))
    (PUBLIC/'calibration_summary.json').write_text(json.dumps(cal_summary, separators=(',', ':'), allow_nan=False)+'\n')
    lines = ['# Neural Fitting Losses', '', 'Fitting objectives are not outer prediction quality or independent risk guarantees.', '',
        '| Head | Updates | First loss | Last logged loss | Fit seconds | Known unique rows |', '|---|---:|---:|---:|---:|---:|']
    for key,r in a['training'].items():
        f = r['fit']
        lines.append(f"| {key} | {f['step']} | {f['trace'][0]['loss']:.8f} | {f['trace'][-1]['loss']:.8f} | {f['seconds']:.3f} | {f['unique_training_rows']} |")
    (PUBLIC/'head_losses.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified=True, policies=72, decisions=216, summary_bytes=(PUBLIC/'summary_metrics.json').stat().st_size)))


if __name__ == '__main__':
    main()
