"""Post-freeze calibration-to-readout harm audit; never selects a policy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.report_m3w_european_cv_reference import sha
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_source_risk_calibration import apply_calibration, calibration_evidence
from src.world_model.m3w_european_source_forecast import baseline_numpy

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'
PACKED = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'


def read(artifact, field, ids):
    path = ROOT/artifact['path']
    if sha(path) != artifact['sha256']:
        raise ValueError('Changed frozen bank')
    with np.load(path, allow_pickle=False) as z:
        pos = np.searchsorted(z['ids'], ids)
        np.testing.assert_array_equal(z['ids'][pos], ids)
        return z[field][pos].copy()


def main():
    analysis = json.loads((PUBLIC/'analysis.json').read_text())
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if not audit['all_passed'] or audit['analysis_sha256'] != sha(PUBLIC/'analysis.json'):
        raise ValueError('Complete independently reconstructed decisions required')
    receipt = json.loads((PACKED/'receipt.json').read_text())
    pid = analysis['identity']['producer_identity']
    if receipt['identity'] != pid['parent_identity']:
        raise ValueError('Changed source identity')
    data = {}
    for name in ('sites', 'history', 'origin', 'target_eval', 'valid', 'baseline_ade'):
        path = PACKED/(name+'.npy')
        if sha(path) != receipt['arrays'][name]:
            raise ValueError('Changed source array')
        data[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    result = []
    for artifact in analysis['calibration']['receipts']:
        path = ROOT/artifact['path']
        if sha(path) != artifact['sha256']:
            raise ValueError('Changed frozen calibration')
        c = json.loads(path.read_text())
        name = path.stem
        candidate, fold, seed, event, _ = name.split('_')
        head = analysis['training'][f'{candidate}_{fold}_{seed}_utility']
        lineage = head['identity']['lineage']
        if c['roles']['fitting'] != lineage['training_sites']:
            raise ValueError('Head fitting/calibration roles differ')
        for role in ('calibration', 'readout'):
            ids = np.flatnonzero(np.isin(data['sites'], c['roles'][role]))
            if candidate == 'damping097':
                pred = baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
                pred = pred.astype(float)+data['origin'][ids, None]
            else:
                pred = read(lineage['final_producer']['prediction'], 'prediction', ids).astype(float)+data['origin'][ids, None]
            ade, _ = native_errors(pred, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            ref = np.asarray(data['baseline_ade'][ids, 1])
            utility = read(c['utility']['scores'], 'scores', ids)
            u = (utility[:, 0]-utility[:, 1])/head['identity']['cost_scale']
            m = read(c['risk']['scores'], 'scores', ids)
            moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
            for rule_name, rule in c['rule']['rules'].items():
                bits = apply_calibration(u, m, moving, rule, .02)
                evidence = calibration_evidence(bits, ref, ade, data['sites'][ids], lineage['easy_cut'], .02)
                result.append(dict(map=name, role=role, rule=rule_name, frozen_rule=rule,
                    indexed_rows=len(ids), switch_rate=float(bits.mean()), **evidence))
    if len(result) != 432:
        raise ValueError('All 72 maps, two roles, three rules required')
    report = dict(analysis_sha256=audit['analysis_sha256'], rows=result,
        result_source='fresh_run_post_freeze_diagnostic_cached_verified_models',
        policy_selected=False, calibrated_guarantee=False, reserved_roles_opened=False)
    (PUBLIC/'calibration_transport.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n')
    lines = ['# Calibration-to-Readout Harm Audit', '',
        'All maps, rules and localities are retained. This is post-freeze diagnosis, not threshold or model selection.',
        'Positive harm counts only increases and cannot cancel against benefits. Net easy degradation is separately reported in results.md.',
        'A 2% positive-harm budget is stricter than 2% net degradation; neither is a physical-safety guarantee.', '',
        '| Map | Role | Rule | Locality | Supported rows | Selected | Net gain (%) | Positive harm/CV (%) | Positive easy harm/easy CV (%) | Zero-CV harmed | Empirical feasible |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in result:
        for site, e in r['by_locality'].items():
            easy = 'undefined' if e['positive_easy_harm_ratio'] is None else f"{100*e['positive_easy_harm_ratio']:.6f}"
            lines.append(f"| {r['map']} | {r['role']} | {r['rule']} | {site} | {e['rows']} | {e['selected']} | {e['net_gain_percent']:.6f} | {100*e['positive_harm_ratio']:.6f} | {easy} | {e['zero_harmed']} | {e['feasible']} |")
    lines += ['', 'Source-development detector tracks only; image pixels, obs8/pred12 rawstride12. No independent certification, metric, seconds, human-gold or physical-safety claim.',
        'No deployment change, Stage5C or SMC.', '']
    (PUBLIC/'calibration_transport.md').write_text('\n'.join(lines))
    print(json.dumps(dict(role_rule_records=len(result), localities=sum(len(r['by_locality']) for r in result), policy_selected=False)))


if __name__ == '__main__':
    main()
