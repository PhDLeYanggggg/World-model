"""Aggregate all registered probability probes without selecting a deployment."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    folder = ROOT/'outputs/publication_readiness_2026_09/motion_start_information'
    report = json.loads((folder/'report.json').read_text())
    replay = json.loads((folder/'replay.json').read_text())
    if (replay['identity'] != report['identity'] or replay['models'] != 48 or
            not all(t['exact'] for t in replay['trials']) or
            replay['report_sha256'] != file_digest(folder/'report.json')):
        raise ValueError('Exact saved-model replay required')
    cache = ROOT/'data/stage_cvpr2027_experiments/stationary_start_probe_v2/fit_rows.npz'
    if file_digest(cache) != report['identity']['stationary_cache_sha256']:
        raise ValueError('Supervisory cache changed')
    with np.load(cache) as a:
        labels = a['targets'].copy(); rows = json.loads(str(a['rows_json']))
    folds = np.array([r['fit_fold'] for r in rows])
    agents = np.array([f"{r['recording_id']}:{r['agent_id']}" for r in rows])
    probabilities = {}
    for t in report['trials']:
        for name, sha in (('model_path', 'model_sha256'), ('prediction_path', 'prediction_sha256')):
            if file_digest(ROOT/t[name]) != t[sha]:
                raise ValueError('Frozen estimator/predictions changed')
        with np.load(ROOT/t['prediction_path']) as a:
            probabilities[t['fold'], t['arm'], t['model'], t['seed']] = a['probability'].copy()
    aggregates = []
    for s in report['summary']:
        fold, arm, model = s['fold'], s['arm'], s['model']
        held = np.flatnonzero(folds == fold)
        seeds = [17, 29, 43]
        p = np.stack([probabilities[fold, arm, model, seed] for seed in seeds])
        q = np.stack([probabilities[fold, 'quality', model, seed] for seed in seeds])
        picked = [t for t in report['trials'] if (t['fold'], t['arm'], t['model']) == (fold, arm, model)]
        prior = picked[0]['held']['train_only_prior']
        losses = ((p-labels[held])**2).mean(0)
        ref_loss = (prior-labels[held])**2
        q_loss = ((q-labels[held])**2).mean(0)
        per_agent = [(float((ref_loss-losses)[agents[held] == g].mean()),
                      float((q_loss-losses)[agents[held] == g].mean()))
                     for g in np.unique(agents[held])]
        aggregates.append(dict(s, held='ETH' if fold == 0 else 'Hotel',
            agents_positive_vs_prior=sum(x[0] > 0 for x in per_agent),
            agents_positive_vs_quality=sum(x[1] > 0 for x in per_agent),
            mean_run_balanced_prior_lift=float(np.mean([t['run_balanced']['brier_lift'] for t in picked])),
            mean_held_log_loss=float(np.mean([t['held']['log_loss'] for t in picked])),
            mean_held_average_precision=float(np.mean([t['held']['average_precision'] for t in picked])),
            mean_held_probability=float(p.mean()), observed_rate=float(labels[held].mean()),
            training_prior=prior))
    result = dict(report_sha256=file_digest(folder/'report.json'),
        analysis_code_sha256=file_digest(Path(__file__)), result_source='fresh_run_aggregate_analysis_cached_verified_models',
        aggregates=aggregates, fit_seconds=sum(t['fit_seconds'] for t in report['trials']),
        fit_warnings=[t['warnings'] for t in report['trials'] if t['warnings']],
        labels_are_annotated_change_not_intent=True, independent_confirmation=False,
        trajectory_improvement_established=False, deployment=False)
    json_write(folder/'analysis.json', result)
    lines = ['# Observed Motion and Start Probability', '',
        '48 fresh classifier fits; no new trajectory or neural model. All values below are seed means.',
        'Brier differences are absolute score differences, not percentages or ADE/FDE gains.', '',
        '| Held site | Model | Inputs | Train Brier lift | Held Brier lift | Held AUC | Lift vs quality | Positive seeds |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for a in aggregates:
        lines.append(f"| {a['held']} | {a['model']} | {a['arm']} | {a['train_brier_lift_mean']:.5f} | "
            f"{a['held_brier_lift_mean']:.5f} | {a['auc_mean']:.4f} | {a['window_lift_over_quality']:.5f} | "
            f"{a['positive_brier_seeds']}/3 |")
    lines += ['', '## Group-Balanced Diagnostic', '',
        'Intervals resample held agents within the same previously exposed site. Five ETH and 26 Hotel agents',
        'do not constitute new-scene confirmation. Seed losses are averaged; probabilities are not ensembled.', '',
        '| Held | Model | Inputs | Agent lift vs prior [95% interval] | Agent lift vs quality [95% interval] | Positive agents vs prior | Run lift vs prior |',
        '| --- | --- | --- | --- | --- | ---: | ---: |']
    for a in aggregates:
        p, q = a['agent_paired_vs_prior'], a['agent_paired_vs_quality']
        def interval(v):
            lo, hi = v['descriptive_ci95']
            return f"{v['agent_balanced_lift']:.5f} [{lo:.5f}, {hi:.5f}]"
        lines.append(f"| {a['held']} | {a['model']} | {a['arm']} | {interval(p)} | {interval(q)} | "
                     f"{a['agents_positive_vs_prior']}/{p['agents']} | {a['mean_run_balanced_prior_lift']:.5f} |")
    lines += ['', 'Zara: not_run, no exactly-static histories. No model/threshold chosen; full forecasting cohort unchanged.',
              'All outputs remain dataset-local/native-step diagnostics. No new deployment, Stage5C or SMC.', '']
    (folder/'results.md').write_text('\n'.join(lines))
    private = ROOT/'data/stage_cvpr2027_experiments/motion_start_information'
    cache_dir = private/'plot_runtime'; cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache_dir); os.environ['XDG_CACHE_HOME'] = str(cache_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'m3w-motion-start-information'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    arms = ['neighbors', 'quality', 'magnitude', 'directed']
    for fold, ax in enumerate(axes):
        for shift, model, color in [(-.18, 'logistic', '#3977a8'), (.18, 'extra_trees', '#b96435')]:
            values = [next(a for a in aggregates if (a['fold'], a['arm'], a['model']) == (fold, arm, model)) for arm in arms]
            ax.bar(np.arange(4)+shift, [a['held_brier_lift_mean'] for a in values], .34, color=color, label=model)
        ax.axhline(0, color='#333333', linewidth=.7)
        ax.set_xticks(range(4), arms, rotation=15); ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('Hotel -> ETH: 5 held agents' if fold == 0 else 'ETH -> Hotel: 26 held agents')
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Brier lift over training-only prior (absolute)')
    fig.suptitle('Observed motion carries a one-direction signal, not robust transfer')
    fig.text(.5, .01, 'Three seed means; overlapping fit windows, not independent confirmation. Trajectory gains are not measured.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(folder/'probability_transfer.svg', metadata={'Date':None})
    fig.savefig(private/'probability_transfer.png', dpi=140); plt.close(fig)
    print(json.dumps(dict(analysis_complete=True, fits=48, fit_seconds=result['fit_seconds'], warnings=result['fit_warnings'])))


if __name__ == '__main__':
    main()
