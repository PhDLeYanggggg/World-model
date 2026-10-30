"""Read complete coupled-head evidence; never use partial policy results for tuning."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_coupled_benefit_risk as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def training():
    reg = json.loads(run.REG.read_text()); freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Scientific registration changed')
    rows = []; states = {}
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if run.sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Trained weights changed')
        state = run.api.parent.read_checkpoint(ROOT/cp['path'])
        if state['step'] != 2000 or state['unknown_rows_with_loss']: raise ValueError('Full label-safe training required')
        states[r['context'], r['seed'], r['mode']] = state; rows.append(r)
    if len(states) != 36 or freeze['registration_sha256'] != run.sha(run.REG): raise ValueError('All36 frozen fits required')
    for c, seed, _ in states:
        run.api.assert_matched(states[c, seed, run.api.ARMS[0]], states[c, seed, run.api.ARMS[1]])
    for a in run.api.ARMS:
        if not json.loads((run.PUBLIC/('first_full_refit_'+a+'.json')).read_text())['scientific_state_exact']:
            raise ValueError('Both complete exact refits required')
    return freeze, rows, states


def training_report():
    freeze, rows, states = training(); stats = {}
    fields = ('total', 'benefit_MSE', 'pointwise_risk', 'selected_query_score')
    for mode in run.api.ARMS:
        selected = [r for r in rows if r['mode'] == mode]
        stats[mode] = {k: dict(initial_median=float(run.np.median([r['first_trace']['monitor'][k] for r in selected])),
            final_median=float(run.np.median([r['final_trace']['monitor'][k] for r in selected])),
            decreased_heads=sum(r['final_trace']['monitor'][k] < r['first_trace']['monitor'][k] for r in selected), required_heads=18) for k in fields}
    lines = ['# Coupled-head training evidence', '', '## Material Passport', '',
        'fresh_run:36 neural fits,72,000 primary updates and two2,000-update complete exact refits.',
        'cached_verified:original input/features,forecasts,normalizers,draw rules and protected model controls.',
        'This report contains fixed fitting monitors,not policy outcomes,validation,test,calibration or deployment evidence.',
        f"PID{freeze['pid']}:{freeze['seconds']:.6f}s from runtime_ready,peakRSS{freeze['peak_RSS_bytes']}B;initial admission is outside that phase timer.",
        'All18 protected model controls are exact;both modes have the same30,857 parameters and draw hashes.', '',
        '|Mode|Fixed fitting monitor|Initial median|Final median|Decreased heads|', '|---|---|---:|---:|---:|']
    for mode in run.api.ARMS:
        for field in fields:
            s = stats[mode][field]; lines.append(f"|{mode}|{field}|{s['initial_median']:.9f}|{s['final_median']:.9f}|{s['decreased_heads']}/18|")
    lines += ['', 'The total is0.5*(benefit+pointwise risk+selected-query score);the protected benefit term is constant with zero gradient.',
        'Do not compare stochastic batch losses as if they were the same fixed examples. Curves below use each head fixed fitting monitor.',
        'No monitor selects a checkpoint or changes settings. All heads complete exactly2,000 updates.',
        '![Fixed fitting monitors](figures/training_monitors.svg)',
        'Unknown outcomes remain context rows and have zero supervised loss. Repeated draws are not independent samples.',
        'This is cost-head training,not a new trajectory forecast/world-dynamics model. No independent roles,Stage5C or SMC.', '']
    run.write(run.PUBLIC/'training_evidence.json', dict(stats=stats, training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        unknown_draws_by_mode={a: sum(r['unknown_rows_drawn'] for r in rows if r['mode'] == a) for a in run.api.ARMS},
        protected_controls_exact=True, policy_outcomes_used=False, independent_confirmation=False))
    (run.PUBLIC/'training_report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6))
    for ax, field in zip(axes, fields):
        for mode, color in zip(run.api.ARMS, ('#557D86', '#AF3D3D')):
            traces = [v['trace'] for (c, seed, a), v in states.items() if a == mode]
            steps = [t['step'] for t in traces[0]]
            if any([r['step'] for r in t] != steps for t in traces): raise ValueError('Monitor step grids differ')
            values = run.np.array([[r['monitor'][field] for r in t] for t in traces])
            ax.plot(steps, run.np.median(values, 0), color=color, label=mode.replace('_benefit', ''))
        ax.set_title(field.replace('_', ' ')); ax.set_xlabel('Actual updates per head'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Fixed exposed-SOURCE fitting monitors,18 heads per arm (not validation)')
    fig.tight_layout(rect=(0, 0, 1, .9)); (run.PUBLIC/'figures').mkdir(exist_ok=True)
    path = run.PUBLIC/'figures/training_monitors.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'training_monitors.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    return stats


def build():
    freeze, _, _ = training(); p = run.PUBLIC
    complete, summary, replay = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not replay['all72groups_exact'] or not replay['all72_original_controls_exact']
            or replay['summary_sha256'] != run.sha(p/'summary.json') or replay['complete_sha256'] != run.sha(p/'complete.json')
            or complete['summary_sha256'] != run.sha(p/'summary.json')
            or complete['training_freeze_sha256'] != run.sha(p/'training_freeze.json')):
        raise ValueError('Complete separate full policy replay required')
    bundle = verify(ROOT, p); groups = aggregate_groups(run.NAME)
    rows = [groups[Path(r['path']).name] for r in complete['groups']]
    s = run.evaluation.summarize(rows, json.loads(run.REG.read_text())['contexts'], json.loads(run.CONFIG.read_text()))
    record_count = len({(r['source'], rec) for r in rows for rec in r['result']['recordings']})
    s.update(unique_target_rows=summary['unique_target_rows'], unique_recordings=record_count)
    if s != summary or record_count != 74 or summary['unique_target_rows'] != 149510: raise ValueError('Canonical full cohort differs')
    old = aggregate_groups(run.unit.NAME)
    for ref, r in zip(complete['groups'], rows):
        prior = old[Path(ref['path']).name]
        if r['result']['policies']['protected_joint'] != prior['result']['policies']['set_joint']:
            raise ValueError('Old complete policy control changed')
    return dict(summary=summary, complete=complete, replay=replay, rows=rows, bundle=bundle,
        training_freeze=freeze, replay_runtime=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    if v['point'] is None or v['CI95'] is None: return 'not_estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]"


def render(v):
    s = v['summary']; lines = ['# Coupled benefit/risk: complete SOURCE development result', '',
        '## Material Passport', '',
        'fresh_run:36 real Torch cost-head fits,72,000 primary updates,two exact full refits,and complete72-group policy readout/replay.',
        'cached_verified:original past inputs,forecasts,floor,normalizers and old protected full-policy choices.',
        'not_run:new trajectory/dynamics training,independent calibration/confirmation,SDD deployment retest.',
        'All72 protected model predictions and full-joint policy controls are exact. Controls are not relabeled independent tests.', '',
        '## Registered Factor', '',
        'Release only the benefit constraint in the selected-set risk recalibrator. Both models have30,857 parameters,',
        'identical initialization/features/draws and2,000 updates. Protected updates reproduce the old model exactly.',
        'Both optimize0.5*(benefit_MSE+pointwise_risk+selected_query_score);benefit_MSE is constant for the protected arm.',
        'Original reference predictions R/easy_R,all observed evaluation targets,strict2% positive-harm/reference budgets and easy preservation remain unchanged.',
        'Primary:released minus protected FULL-joint completion-lower utility. Matched-count is a secondary utility control,not a safety replacement.', '',
        '|Released minus protected|Known pp [95% CI]|Completion-lower pp [95% CI]|', '|---|---:|---:|']
    for mode in ('joint', 'matched'):
        lines.append(f"|{mode}|{fmt(s['effects'][mode]['known_pp'])}|{fmt(s['effects'][mode]['lower_pp'])}|")
    lines += ['', '|Policy|Known ADE gain over floor (%)|Full-roster FDE|Supported original record-risk views|Known easy-risk violations|Undefined easy|Selected / unknown|',
        '|---|---:|---|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        risk = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{fmt(s['FDE'][a])}|{risk['supported_record_views']}/{risk['required_record_views']}|"
            f"{risk['known_easy_risk_violations']}|{risk['undefined_easy_record_views']}|{risk['selected']}/{risk['selected_unknown']}|")
    lines += ['', 'Genuine positive-harm/reference risk is not net error;benefit never cancels its numerator.',
        'All149,510 unique target rows,74 recordings,12 exposed localities and444 reused record/context/critic-seed views remain.',
        'Wholly unknown outcomes remain in bounds and are not safe zeros. Empty matched risk is not_estimable.',
        'Unavailable required endpoints keep full-roster FDE not_estimable;available errors remain supplementary.', '',
        '## Fitting Checks', '']
    lines += [f"- {k}: {'pass' if b else 'fail'}" for k, b in s['fitting_gates'].items()]
    lines += ['', f"Advancement:{s['fitting_advancement_gate']}. Deployment unchanged,even if an exploratory fitting check passes.", '',
        '## Limits', '',
        'Intervals are nominal3,000-draw locality bootstraps after averaging reused contexts and critic seeds within12 EXPOSED SOURCE localities.',
        'Not independent confirmation,not whole-pipeline OOF,and not adjusted for adaptive prior research. Three critic seeds share forecast producer43.',
        'Any paired cost-head effect includes changes to predicted utility and admissible harm capacity;it is not isolated proof of physical dynamics.',
        'Record tails,hard/easy pooled known masses,and prediction-only pair-distance contraction are in complete aggregate group receipts.',
        'Pair proxies cover query targets,not all visible agents,and do not certify physical collision safety.',
        'A lower fitting monitor does not establish decision calibration,conformal control or deployment.',
        'Obs8/pred12/rawstride12,oblique detector-silver/local-image pixels,not SDDt50,metric,seconds,human gold,true3D or foundation.',
        'No independent role,Stage5C execution or SMC. Not submission-ready.', '',
        '[Training curves](training_report.md),[method and primary-source compatibility](method_note.md),',
        '[frozen protocol](protocol.md),[failure analysis](failure_analysis.md),[reproduction](reproduction_zh.md).', '']
    return '\n'.join(lines)


def report():
    training_report(); v = build(); p = run.PUBLIC
    (p/'readout.md').write_text(render(v))
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), training_freeze_sha256=run.sha(p/'training_freeze.json'),
        report_code_sha256=run.sha(Path(__file__)), readout_sha256=run.sha(p/'readout.md'),
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(complete_report=True, fitting_gates=v['summary']['fitting_gates'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--training-only', action='store_true'); a = p.parse_args()
    training_report() if a.training_only else report()
