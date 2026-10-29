"""Verified complete cost-retrieval evidence, without publishing private donors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_coupled_local_cost as run


def interval(value):
    if value['point'] is None:
        return 'not_estimable'
    if value['CI95'] is None:
        return f"{value['point']:+.6f} [not_estimable]"
    return f"{value['point']:+.6f} [{value['CI95'][0]:+.6f}, {value['CI95'][1]:+.6f}]"


def build():
    p = run.PUBLIC; summary = json.loads((p/'summary.json').read_text())
    complete = json.loads((p/'complete.json').read_text()); check = json.loads((p/'verification.json').read_text())
    if (not check['exact_replay'] or check['groups'] != 288 or len(complete['groups']) != 288
            or check['summary_sha256'] != run.sha(p/'summary.json')
            or summary['held_scored'] or summary['independent_roles_opened']):
        raise ValueError('Complete exact fitting replay required')
    rows = []
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Completed metric group changed')
        rows.append(json.loads(path.read_text()))
    modes = {}
    for mode in run.api.MODES:
        group = [r for r in rows if r['mode'] == mode]
        if len(group) != 72:
            raise ValueError('Every original source/control-seed group required')
        modes[mode] = dict(unique_fitting_recordings=len({(r['source'],k) for r in group for k in r['result']['recordings']}),
            query_occurrences=sum(r['result']['rows'] for r in group),
            unknown_query_occurrences=sum(r['result']['unknown_rows'] for r in group),
            proxy_fit={})
        for bank in run.parent.api.BANKS:
            modes[mode]['proxy_fit'][bank] = {}
            for subset in ('all','known','unknown'):
                values = [r['result']['proxy_diagnostics'][bank]['finite_endpoint_proxy_MSE'][subset] for r in group]
                n = sum(v['real_action_rows'] for v in values)
                error = sum((run.np.asarray(v['component_MSE'])*v['real_action_rows'] for v in values if v['component_MSE'] is not None),run.np.zeros(5))
                modes[mode]['proxy_fit'][bank][subset] = dict(real_action_occurrences=n,
                    component_MSE=(error/n).tolist() if n else None,
                    undefined_groups=sum(v['component_MSE'] is None for v in values),
                    aggregation='pooled_repeated_real_action_views_not_macro_or_independent')
    return dict(primary_summary_sha256=run.sha(p/'summary.json'),verification_sha256=run.sha(p/'verification.json'),
        result_source='fresh_aggregate_report_of_exact_fitting_control',modes=modes,
        compute={k:complete[k] for k in ('pid','seconds','peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()),
        gates=summary['fitting_gates'],fitting_advancement_gate=summary['fitting_advancement_gate'],
        new_optimizer_updates=0,held_scored=False,independent_roles_opened=False,
        whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def render(evidence):
    summary = json.loads((run.PUBLIC/'summary.json').read_text()); c=evidence['compute']; v=evidence['replay']
    lines=['# Coupled Local Cost Retrieval: Complete Fitting Results','','## Material Passport','',
        'Fresh_run: exact other-recording retrieval, 288 comparison groups and separate',
        'full retrieval/readout replay. Cached_verified: original source packets, forecast',
        'producers and the paired cost-head controls. Not_run: new neural optimization,',
        'trajectory training, independent validation/confirmation or deployment.','',
        f"First PID {c['pid']}, exit0, {c['seconds']:.6f}s, peak RSS {c['peak_RSS_bytes']:,} B.",
        f"Replay PID {v['pid']}, {v['seconds']:.6f}s, peak RSS {v['peak_RSS_bytes']:,} B.",
        'Native arm64 CPU4, workers0, checkpointed neighbors and private heartbeat.',
        'The new estimator is deterministic: repeated comparator seeds are not three',
        'independent retrieval models or independently trained forecasting pipelines.','',
        '## Fixed Comparison','',
        'All queries and original labels remain. Each query excludes its entire recording',
        'from fitted donors. The fixed 398-field distance descriptor averages standardized',
        'physical-action context. Shared-reference and convex harm decoding preserve',
        'moment ordering, but do not certify selected-risk calibration. Normalizers and',
        'upstream producers saw fitting data, so this is not whole-pipeline OOF.','',
        '## Accuracy And Full Risk','',
        '| Mode | Bank | Known ADE gain over floor (%), fitting CI | Required risk support | Undefined easy | Known all/easy violations | Selected (unknown) |',
        '|---|---|---|---:|---:|---:|---:|']
    for mode,data in summary['modes'].items():
        for bank in run.parent.api.BANKS:
            s=data['safety'][bank]
            lines.append(f"| {mode} | {bank} | {interval(data['accuracy'][bank])} | {s['supported_record_views']}/{s['required_record_views']} | "
                f"{s['undefined_easy_record_views']} | {s['known_all_risk_violations']}/{s['known_easy_risk_violations']} | {s['selected']} ({s['selected_unknown']}) |")
    lines += ['','No empty selected-risk estimate passes by default. All required recordings,',
        'unknown outcomes and undefined ratios remain. Count matching is utility only.','',
        '## Mixed Policy Minus Frozen Known-Only Cost Head','',
        '| Mode | Full known change (pp), fitting CI | Full completion-lower (pp), fitting CI | Matched completion-lower (pp), fitting CI |',
        '|---|---|---|---|']
    for mode,data in summary['modes'].items():
        d=data['prior_known_change'];lines.append('| '+mode+' | '+' | '.join(interval(d[k]) for k in ('full_known_pp','full_lower_pp','matched_lower_pp'))+' |')
    lines += ['','## Mixed Minus Physical Bank','',
        '| Mode | Full known increment (pp), fitting CI | Full completion-lower (pp), fitting CI |',
        '|---|---|---|']
    for mode,data in summary['modes'].items():
        d=data['comparisons']['physical_neural_minus_physical_only']
        lines.append('| '+mode+' | '+interval(d['full_known_pp'])+' | '+interval(d['full_lower_pp'])+' |')
    d=summary['local_over_global']
    lines += ['','## Primary Local-Bounds Minus Global-Bounds','',
        '- Full known: '+interval(d['full_known_pp'])+' pp.',
        '- Full completion-lower: '+interval(d['full_lower_pp'])+' pp.',
        '- Matched completion-lower: '+interval(d['matched_lower_pp'])+' pp.','',
        '## Known Hard/Easy Slices','',
        'Supplementary pooled errors over repeated views, not source-macro accuracy,',
        'independent sample counts or substitutes for full recording risk.','',
        '| Mode | Subset | Known row views | Empty source views | Mixed ADE gain (%) | Mixed FDE gain (%) |',
        '|---|---|---:|---:|---:|---:|']
    for mode,data in summary['modes'].items():
        for subset,s in data['subsets'].items():
            p=s['policies']['physical_neural']
            show=lambda x:'not_estimable' if x is None else f'{x:+.6f}'
            lines.append(f"| {mode} | {subset} | {s['known_row_views']} | {s['empty_known_views']} | {show(p['ADE_gain_percent'])} | {show(p['FDE_gain_percent'])} |")
    lines += ['','## Proxy Fit','',
        'Normalized component MSE uses only real candidate actions. Unknown targets',
        'are interval endpoints, not actual future errors or a calibration certificate.','',
        '| Mode | Mixed-bank support | Real action occurrences | Undefined groups | B / H / R / ER / EH MSE |',
        '|---|---|---:|---:|---|']
    for mode,data in evidence['modes'].items():
        for subset,p in data['proxy_fit']['physical_neural'].items():
            value='not_estimable' if p['component_MSE'] is None else ', '.join(f'{x:.6g}' for x in p['component_MSE'])
            lines.append(f"| {mode} | {subset} | {p['real_action_occurrences']} | {p['undefined_groups']} | {value} |")
    lines += ['','## Gates','',*['- '+k+': '+json.dumps(v) for k,v in evidence['gates'].items()],
        '- fitting_advancement_gate: '+json.dumps(evidence['fitting_advancement_gate']),'',
        'The primary and all secondary controls remain visible; no better-looking',
        'secondary is promoted after readout. All intervals are 3,000 fitting-locality',
        'bootstrap draws after dependent context/comparator-seed averaging. They are',
        'not independent generalization, simultaneous certification or correction for',
        'repeated research selection. Undefined required FDE remains not_estimable.','',
        'SOURCE is oblique detector-track midpoint pixels, obs8/pred12/stride12 raw',
        'frames. No metric/seconds, human-gold, true-3D, foundation, physical-safety or',
        'new world-dynamics claim. Deployment, independent roles, Stage5C and SMC are',
        'unchanged. See failure_analysis.md and reproduction_zh.md for interpretation',
        'and verified command scope.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    value=build();run.once(run.PUBLIC/'evidence.json',value);text=render(value);p=run.PUBLIC/'readout.md'
    if p.exists() and p.read_text()!=text:
        raise ValueError('Frozen report changed')
    p.write_text(text);print(json.dumps(dict(fitting_advancement_gate=value['fitting_advancement_gate'])))
