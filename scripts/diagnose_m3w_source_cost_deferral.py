"""Post-hoc training-only score/target diagnosis; never fits a deployment policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_cost_deferral import load_config, build_data
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def huber_location(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite one-dimensional signed training costs required')
    low, high = float(values.min()), float(values.max())
    for _ in range(100):
        middle = (low + high)/2
        if np.clip(middle-values, -1, 1).mean() < 0:
            low = middle
        else:
            high = middle
    value = (low + high)/2
    assert abs(np.clip(value-values, -1, 1).mean()) < 1e-10
    return value


def summarize(gain, score, p, choose):
    gain, score = np.asarray(gain, float), np.asarray(score, float)
    return dict(rows=len(gain), helpful_rows=int((gain > 0).sum()),
        harmful_rows=int((gain < 0).sum()), mean_gain=float(gain.mean()),
        median_gain=float(np.median(gain)), huber_location=huber_location(gain),
        gain_quantiles=np.quantile(gain, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        outside_unit_huber_radius=int((np.abs(gain) > 1).sum()),
        mean_positive_gain=float(np.maximum(gain, 0).mean()),
        mean_negative_gain=float(np.minimum(gain, 0).mean()),
        score_quantiles=np.quantile(score, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        requested_rate=float(choose.mean()), mean_score=float(score.mean()),
        mean_soft_gate=float(p.mean()), hard_gain=float((choose*gain).mean()),
        soft_expected_gain=float((p*gain).mean()),
        signed_cost_rmse=float(np.sqrt(np.mean((score-gain)**2))),
        constant_training_mean_rmse=float(np.std(gain)),
        zero_score_rmse=float(np.sqrt(np.mean(gain**2))),
        expected_risk_score_gradient=float(np.mean(-p*(1-p)*gain)),
        huber_score_gradient=float(np.clip(score-gain, -1, 1).mean()),
        hard_helpful_rows=int((choose & (gain > 0)).sum()),
        hard_harmful_rows=int((choose & (gain < 0)).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public = ROOT/reg['reports']
    report = json.loads((public/'report.json').read_text())
    analysis = json.loads((public/'analysis.json').read_text())
    assert analysis['report_sha256'] == file_digest(public/'report.json')
    assert analysis['exact_milestone_replays'] == 24 and analysis['held_rows_scored'] == 0
    data, ids, _, scale, _, controls = build_data(reg)
    loc = ids-data.nmain
    target = data.target[loc].astype(float)
    baseline = np.linalg.norm(target, axis=-1).mean(1)
    hard = baseline >= analysis['training_hard_cut']
    masks = dict(all=np.ones(len(ids), bool), zero_target=baseline == 0,
                 nonzero_target=baseline > 0, hard=hard)
    rows = []
    hashes = {str((public/'report.json').relative_to(ROOT)):file_digest(public/'report.json'),
              str((public/'analysis.json').relative_to(ROOT)):file_digest(public/'analysis.json')}
    for trial in report['trials']:
        m = trial['milestones'][-1]
        path = ROOT/m['prediction_path']
        assert file_digest(path) == m['prediction_sha256'] and m['step'] == 10000
        hashes[m['prediction_path']] = m['prediction_sha256']
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], ids)
            proposal, score = a['proposal'].astype(float), a['score'].copy()
        error = np.linalg.norm(proposal-target, axis=-1).mean(1)
        gain = (baseline-error)/scale
        p = torch.from_numpy(score).sigmoid().numpy()
        for name, mask in masks.items():
            rows.append(dict(trial=trial['trial'], variant=trial['variant'],
                seed=trial['seed'], slice=name, **summarize(gain[mask], score[mask], p[mask], score[mask] > 0)))
    result = dict(result_source='fresh_run_posthoc_training_diagnostic',
        registered_before_training=False, model_or_threshold_changed=False,
        training_rows=len(ids), held_rows_scored=0, main_rows_scored=0,
        inputs_sha256=hashes, rows=rows,
        interpretation='normalized signed gain per row, not a percent or calibrated risk',
        covariance_or_conditional_feature_claim=False, new_deployment=False)
    json_write(public/'score_target_diagnostic.json', result)
    lines = ['# Post-Hoc Training Score and Target Diagnosis', '',
        'Defined after the first seed results were observed. No training, threshold search,',
        'held scoring or deployment is performed by this analysis. All six fixed endpoints',
        'are retained. Costs below are normalized signed gain, not percentage gains.', '',
        '| Branch | Mean target | Median target | Huber location | Mean score | RMSE / constant mean RMSE | Outside unit Huber radius | Requested rate |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['slice'] != 'all':
            continue
        lines.append(f"| {r['trial']} | {r['mean_gain']:+.7f} | {r['median_gain']:+.7f} | {r['huber_location']:+.7f} | {r['mean_score']:+.7f} | {r['signed_cost_rmse']:.7f} / {r['constant_training_mean_rmse']:.7f} | {r['outside_unit_huber_radius']} | {r['requested_rate']:.3%} |")
    lines += ['', '## What This Can and Cannot Establish', '',
        'For a fixed candidate, expected-action cost has score derivative',
        '`-sigmoid(z)*(1-sigmoid(z))*gain`. The Smooth-L1 term has derivative',
        '`clip(z-gain, -1, 1)`. Its constant optimum is a Huber location, not',
        'necessarily the arithmetic mean gain that determines aggregate expected cost.',
        'The comparison above checks whether this distinction is material for the',
        'observed training targets; it does not prove the conditional optimum of a neural head.', '',
        'The expected-cost branch has no gain-regression supervision, so its signed-cost',
        'RMSE is descriptive only, not a failed promised calibration objective.',
        'The cost-supervised branch jointly changes its candidate and representation;',
        'in-sample score fit does not establish independent calibration or generalization.', '',
        'Gradient averages concern a hypothetical common score shift with the candidate',
        'held fixed. They are not parameter gradients, a convergence certificate or an',
        'isolated explanation of clipping. Slice targets are future-label diagnostics',
        'and never inference features. No future/test data informs a policy change.', '',
        'Past-normalized/pixel raw-frame source task only. No Stage5C execution, SMC,',
        'metric/seconds, true-3D, foundation or submission-ready claim.', '']
    (public/'score_target_diagnostic.md').write_text('\n'.join(lines))
    print(json.dumps([r for r in rows if r['slice'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
