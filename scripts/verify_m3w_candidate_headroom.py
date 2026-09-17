"""Independent SciPy scalar-minimization check of real diagnostic paths."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    reports = ROOT/'outputs/publication_readiness_2026_09/candidate_headroom'
    report = json.loads((reports/'report.json').read_text())
    replay = json.loads((reports/'replay.json').read_text())
    reg_path = ROOT/'configs/m3w_candidate_headroom.json'
    if report['identity']['registration_sha256'] != file_digest(reg_path) or replay['recomputed'] != 72:
        raise ValueError('Require unchanged registration and full real replay')
    reg = json.loads(reg_path.read_text())
    frozen_path = ROOT/reg['frozen_report']
    if file_digest(frozen_path) != report['identity']['frozen_report_sha256']:
        raise ValueError('Frozen prediction report changed')
    frozen = json.loads(frozen_path.read_text())
    source = ROOT/'data/stage_cvpr2027_experiments/offline_visual_forecast/inputs'
    manifest = json.loads((source/'data_manifest.json').read_text())
    if file_digest(source/'data_manifest.json') != report['identity']['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name in ('targets.npy', 'baselines.npy'):
        if file_digest(source/name) != manifest['arrays'][name]:
            raise ValueError('Source array changed')
    targets = np.load(source/'targets.npy', mmap_mode='r')
    base = np.load(source/'baselines.npy', mmap_mode='r')[:, manifest['baseline_names'].index('constant_velocity_causal_fd')]
    receipts = {r['identity']['trial']:r for r in report['receipts']}
    checks, maximum_difference, maximum_relative = 0, 0., 0.
    for trial_number, t in enumerate(frozen['trials']):
        prediction_path = ROOT/t['prediction_path']
        oracle_path = ROOT/reg['output']/(t['trial']+'.npz')
        if file_digest(prediction_path) != t['prediction_sha256'] or file_digest(oracle_path) != receipts[t['trial']]['sha256']:
            raise ValueError('Changed real diagnostic artifact')
        with np.load(prediction_path) as pred, np.load(oracle_path) as oracle:
            indices = np.random.default_rng(2026+trial_number).choice(len(pred['held_indices']), 12, replace=False)
            for i in indices:
                source_row = int(pred['held_indices'][i])
                a, b, y = base[source_row].astype(float), pred['prediction'][i].astype(float), targets[source_row].astype(float)
                def objective(alpha):
                    return float(np.linalg.norm(a+alpha*(b-a)-y, axis=-1).mean())
                result = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol':1e-12, 'maxiter':150})
                if not result.success:
                    raise ValueError('Independent minimizer did not converge')
                scipy_loss = min(objective(0), objective(1), result.fun)
                stored, lower = float(oracle['feasible_loss'][i]), float(oracle['lower_loss'][i])
                tolerance = 1e-7*max(1., scipy_loss)
                if abs(stored-scipy_loss) > tolerance or lower > scipy_loss+tolerance:
                    raise ValueError('Independent scalar minimizer disagrees')
                maximum_difference = max(maximum_difference, abs(stored-scipy_loss))
                maximum_relative = max(maximum_relative, abs(stored-scipy_loss)/max(1., scipy_loss))
                checks += 1
    import scipy
    result = dict(result_source='fresh_run_independent_scipy_path_checks', real_paths=checks,
        scipy_version=scipy.__version__, maximum_absolute_loss_difference=maximum_difference,
        maximum_difference_divided_by_max1_loss=maximum_relative, relative_tolerance=1e-7,
        report_sha256=file_digest(reports/'report.json'), verification_code_sha256=file_digest(Path(__file__)),
        exact_recomputation_trials=72, not_interval_arithmetic_certification=True,
        model_training_or_selection_run=False, sealed_roles_opened=False)
    json_write(reports/'independent_verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
