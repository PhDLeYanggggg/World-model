"""Verify and render complete online-admission evidence without rescoring outcomes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_admission_cost as run
from scripts.run_m3w_selected_action_attribution import pool
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def build():
    public = run.PUBLIC
    summary = json.loads((public / 'summary.json').read_text())
    complete = json.loads((public / 'complete.json').read_text())
    verification = json.loads((public / 'verification.json').read_text())
    freeze = json.loads((public / 'training_freeze.json').read_text())
    if (verification['summary_sha256'] != run.sha(public / 'summary.json')
            or verification['complete_sha256'] != run.sha(public / 'complete.json')
            or not verification['all72groups_exact'] or not verification['all144control_predictions_actions_full_metrics_exact']
            or freeze['count'] != 54 or freeze['exact_parent_controls'] != 36
            or complete['summary_sha256'] != run.sha(public / 'summary.json')):
        raise ValueError('Complete exact verified scientific evidence required')
    models = run.models()
    rows = list(aggregate_groups(run.NAME).values())
    if len(rows) != 72:
        raise ValueError('Complete72group roster required')
    attribution = {o: pool([v['result']['attribution'][o] for v in rows]) for o in run.api.OBJECTIVES}
    for filename in ('registration.json', 'readout_registration.json'):
        for rel, digest in json.loads((public / filename).read_text())['bindings'].items():
            if run.sha(ROOT / rel) != digest:
                raise ValueError('Frozen scientific implementation changed: ' + rel)
    first = sorted(json.loads(run.REG.read_text())['contexts'])[0]
    refit = json.loads((public / 'first_full_refit.json').read_text())
    if (not refit['scientific_state_exact']
            or run.sha(run.checkpoint(first, 17, 'admitted_score')) != refit['checkpoint_sha256']
            or run.sha(run.checkpoint(first, 17, 'admitted_score', 'full_refit')) != refit['refit_checkpoint_sha256']):
        raise ValueError('Full refit identity changed')
    traces = {}
    for key, row in sorted(models.items()):
        state = run.parent.frozen.api.completion.base.read_checkpoint(ROOT / row['checkpoint']['path'])
        if state['step'] != 2000 or state['unknown_rows_with_loss'] or state['parameters'] != 30405:
            raise ValueError('Real full-budget known-only models required')
        traces['|'.join(map(str, key))] = state['trace']
    return dict(summary=summary, attribution=attribution, training=freeze, readout=complete,
        verification_runtime=json.loads((run.PRIVATE / 'verification_runtime.json').read_text()),
        checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in models.values()),
        traces=traces, groups=rows)


def fmt(v):
    if v['point'] is None:
        return 'not estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def render(value):
    s, a = value['summary'], value['attribution']
    train, read = value['training'], value['readout']
    gates = s['fitting_gates']
    lines = ['# Online selected-action loss: complete readout', '',
        '## Material Passport', '',
        'Mode: experiment execution and verification. Data role: exposed SOURCE fitting only.',
        'Result source: fresh native arm64 Torch training and inference; cached verified inputs and trajectory banks.',
        'Status: complete and separately replayed exactly. Independent confirmation: not run.', '',
        '## What changed', '',
        'I added one unit-weight loss on the currently chosen supervised false-admitted action.',
        'The full upper-score/five-moment/ranking loss remains. Inputs, width, initialization,',
        'sampling, normalization, proposal bank, decoder and original 2% risk rule do not change.',
        'This is a cost-head learning experiment, not a newly trained trajectory producer.', '',
        f"All54heads complete2000updates:108000primary updates,2000full-refit updates and160pilot updates.",
        f"All36original/upper full scientific states match; the admitted head full refit is exact.",
        f"Training PID{train['pid']} took{train['seconds']:.3f}s, peak RSS{train['peak_RSS_bytes']}B.",
        f"Readout PID{read['pid']} took{read['seconds']:.3f}s, peak RSS{read['peak_RSS_bytes']}B.",
        f"All72groups and144control prediction/action/full-metric checks agree in a separate replay.",
        f"Private primary checkpoint size:{value['checkpoint_bytes']}B. All54actual traces accompany this report.", '',
        '## Complete-policy results', '',
        'Known ADE gain versus the frozen floor; percentage points, equal locality, 95% bootstrap interval.',
        'Three controller seeds,3000resamples over12localities after context/seed averaging.', '',
        '|Objective|Known ADE gain|Supported recording views|Known easy risk violations|Known all risk violations|Undefined easy views|',
        '|---|---:|---:|---:|---:|---:|']
    for o in run.api.OBJECTIVES:
        r = s['safety'][o]
        lines.append(f"|{o}|{fmt(s['accuracy'][o])}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_easy_risk_violations']}|{r['known_all_risk_violations']}|{r['undefined_easy_record_views']}|")
    lines += ['', '## Fixed contrasts', '', '|Contrast|Known pp [95% CI]|Completion-lower pp [95% CI]|', '|---|---:|---:|']
    for name, item in s['comparisons'].items():
        lines.append(f"|{name}|{fmt(item['full_known_pp'])}|{fmt(item['full_lower_pp'])}|")
    lines.append(f"|Same admitted head actual neural use|{fmt(s['actual_neural_use']['full_known_pp'])}|{fmt(s['actual_neural_use']['full_lower_pp'])}|")
    lines += ['', 'Completion sensitivity varies wholly unknown-row policy disagreement only. Partial observed',
        'ADE remains fixed at its original available-point estimand; it is not full12-step recovery.',
        'The complete-roster FDE is not estimable where a required locality lacks endpoints.',
        'Same-count comparisons are utility-only; empty is not estimable, never safe zero.', '',
        '## Selection failure mechanism', '',
        '|Objective|Selected known|Selected unknown|Easy false admissions|Easy numerator-only|Observed easy harm|Predicted easy harm|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for o in run.api.OBJECTIVES:
        ch = a[o]['channels']['easy']
        lines.append(f"|{o}|{a[o]['selected_known_rows']}|{a[o]['selected_unknown_rows']}|"
            f"{ch['false_admitted_known_rows']}|{ch['witness_counts']['numerator_only']}|"
            f"{ch['observed_harm_mass']:.6f}|{ch['predicted_harm_mass']:.6f}|")
    lines += ['', 'Arithmetic witnesses use future evaluation labels only, never deployment rules or causal proof.',
        'Pooled counts and masses are overlapping fitting views, not independent samples or metric quantities.',
        'Support remains149510unique queries,74recordings,12localities; no test/independent outcome opens.', '',
        '## Gates', '']
    lines += [f"- {name}: {'pass' if passed else 'fail'}" for name, passed in gates.items()]
    lines += ['', f"Fitting-progress gates:{sum(gates.values())}/{len(gates)}. Complete advancement:{s['fitting_advancement_gate']}.",
        'No independent confirmation, deployment change or submission-ready claim follows from fitting gates.', '',
        '## Limitations and next action', '',
        'The global2% selected-reference rule remains mandatory in every required recording. Positive',
        'utility, fewer unsafe rows or positive actual-neural-use contrast cannot replace that rule.',
        'This experiment uses exposed development data; all intervals are conditional and exploratory.',
        'If recording risk or primary utility still fails, retain the fallback and diagnose the changed',
        'selected population before another distinct repair. Do not tune on reserved outcomes.',
        'Independent calibration/confirmation and a defensible primary contribution remain required.',
        'This supplied detector-silver top-down task remains pixel/raw-frame, not metric, seconds-level,',
        'human gold, true3D or foundation world modeling. No Stage5C execution or SMC.', '']
    return '\n'.join(lines)


def figures(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                        'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    s = value['summary']; labels = ['Original', 'Upper score', 'Online admission']
    objects = run.api.OBJECTIVES
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, (o, color) in enumerate(zip(objects, ('#555555', '#007C91', '#8B3F72'))):
        v = s['accuracy'][o]; low, high = v['CI95']
        ax[0].errorbar(v['point'], i, xerr=[[v['point']-low], [high-v['point']]],
            color=color, fmt='o', capsize=4)
    ax[0].set_yticks(range(3), labels)
    ax[0].set_xlabel('Known ADE gain (%) vs frozen floor')
    ax[0].set_title('Fitting utility: locality bootstrap95% CI', fontsize=11)
    ax[0].invert_yaxis(); ax[0].grid(axis='x', alpha=.2)
    metrics = ('supported_record_views', 'known_easy_risk_violations', 'undefined_easy_record_views')
    x = np.arange(3)
    for j, (key, label, color) in enumerate(zip(metrics,
            ('Supported', 'Known easy violation', 'Undefined easy'), ('#007C91', '#B44D48', '#D4A43A'))):
        ax[1].bar(x + (j-1)*.24, [s['safety'][o][key] for o in objects], width=.23, label=label, color=color)
    ax[1].set_xticks(x, labels); ax[1].set_ylim(0, 480)
    ax[1].set_ylabel('Recording/context/seed views (444 required)')
    ax[1].set_title('Complete-policy risk support, not mean net gain', fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('Exposed SOURCE fitting only; no independent or deployment promotion', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    folder = run.PUBLIC / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'admission_summary.svg', bbox_inches='tight', metadata={'Date': None})
    fig.savefig(run.PRIVATE / 'admission_summary.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    value = build()
    run.write(run.PUBLIC / 'readout.md', render(value))
    run.write(run.PUBLIC / 'training_traces.json', dict(heads=54, actual_optimizer_traces=value['traces']))
    figures(value)
    run.write(run.PUBLIC / 'evidence.json', dict(
        verified_summary_sha256=run.sha(run.PUBLIC / 'summary.json'),
        verified_complete_sha256=run.sha(run.PUBLIC / 'complete.json'),
        verification_sha256=run.sha(run.PUBLIC / 'verification.json'),
        report_code_sha256=run.sha(Path(__file__)),
        readout_sha256=run.sha(run.PUBLIC / 'readout.md'),
        traces_sha256=run.sha(run.PUBLIC / 'training_traces.json'),
        figure_sha256=run.sha(run.PUBLIC / 'figures/admission_summary.svg'),
        all54weights_and72aggregate_groups_verified=True,
        scientific_rescoring=False, independent_confirmation=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(report_generated=True, gates=value['summary']['fitting_gates'],
        checkpoint_bytes=value['checkpoint_bytes'])))


if __name__ == '__main__':
    main()
