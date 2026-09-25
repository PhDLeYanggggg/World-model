"""Verified producer-bank completion, explicitly separate from model performance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_nested_calibration_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    replay = json.loads((PUBLIC/'producer_replay.json').read_text())
    if not replay['all_passed'] or replay['predictors'] != 18 or len(replay['checks']) != 18:
        raise ValueError('Complete producer-bank replay required')
    rows = []
    for item in replay['checks']:
        for field in ('prediction', 'checkpoint'):
            artifact = item[field]
            if sha(ROOT/artifact['path']) != artifact['sha256']:
                raise ValueError('Producer artifacts changed')
        if not item['exact']:
            raise ValueError('Exact inference replay required')
        r = json.loads((PRIVATE/'producers'/item['trial']/'complete.json').read_text())
        if r['identity']['identity'] != replay['identity'] or r['checkpoint'] != item['checkpoint']:
            raise ValueError('Producer lineage changed')
        f = r['fit']
        if f['step'] != 4000 or f['total_draws'] != 256000 or f['held_rows_sampled']:
            raise ValueError('Invalid producer budget or exposure')
        rows.append(dict(trial=item['trial'], training_sites=r['identity']['training_sites'],
            predicted_sites=r['identity']['predicted_sites'], updates=f['step'], fit_seconds=f['seconds'],
            total_draws=f['total_draws'], held_rows_sampled=f['held_rows_sampled'],
            unknown_zero_loss_draws=r['unknown_zero_loss_draws'], unique_training_rows=f['unique_training_rows'],
            first_loss=f['losses'][0]['loss'], last_loss=f['losses'][-1]['loss'], parameters=f['parameters'],
            prediction_rows=item['prediction_rows'], replay_rows=item['replay_rows'], artifacts={
                field:item[field] for field in ('prediction','checkpoint')}))
    result = dict(result_source='fresh_run_inner_predictor_training_cached_verified_final_predictors',
        identity=replay['identity'], predictors=rows, new_updates=sum(r['updates'] for r in rows),
        cached_final_predictors=9, exact_replay=True,
        total_fit_seconds=sum(r['fit_seconds'] for r in rows),
        calibration_and_outer_error_readout_in_this_phase=False,
        scientific_safety_repair_proven=False, reserved_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    (PUBLIC/'producer_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Calibration-Excluded Producer Bank', '',
        '18 new Torch predictors / 72,000 updates. Nine frozen final four-locality producers are cached_verified.',
        'All opposite-half prediction banks replay exactly. This report establishes an executable, excluded producer chain, not downstream improvement.', '',
        '| Predictor | Fitting sites | Opposite predicted sites | Prediction rows | First loss | Last logged loss | Fit seconds | Unknown zero-loss draws |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['trial']} | {', '.join(r['training_sites'])} | {', '.join(r['predicted_sites'])} | {r['prediction_rows']} | {r['first_loss']:.6f} | {r['last_loss']:.6f} | {r['fit_seconds']:.3f} | {r['unknown_zero_loss_draws']} |")
    lines += ['', f"Summed fit-loop time: {result['total_fit_seconds']:.3f} seconds, excluding input/hash checks, prediction and replay.",
        'Each predictor has 256,000 draws; no calibration, outer or opposite-half row is sampled in its fit.',
        'Unknown future labels contribute zero supervised loss under the unchanged parent sampler; they are not claimed as labeled examples.',
        'Inference is past geometry only. Internal producer sizes differ from the final four-locality predictor; downstream calibration compares matched new chains.',
        'Three seeds and both future role rotations are retained. Training loss does not prove held-out prediction or safety gains.',
        'This producer-bank phase does not read calibration or outer trajectory errors; those belong to subsequent frozen head/calibration phases.',
        'Source-development detector tracks, image pixels, obs8/pred12 rawstride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claim.',
        'No deployment promotion, Stage5C or SMC. No independent reserved role opened.', '']
    (PUBLIC/'producer_report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(predictors=18, updates=72000, all_replayed=True, downstream_gain_proven=False)))


if __name__ == '__main__':
    main()
