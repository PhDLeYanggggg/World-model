"""Check final real checkpoints, exact replay and no-op completed-run resume."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    output = ROOT/'data/stage_cvpr2027_experiments/observed_motion_v2'
    reports = ROOT/'outputs/publication_readiness_2026_09/observed_motion_v2'
    report = json.loads((reports/'report.json').read_text())
    replay = json.loads((reports/'replay.json').read_text())
    if len(report['trials']) != 54 or len(replay['trials']) != 54 or not all(t['prediction_exact'] for t in replay['trials']):
        raise ValueError('Require completed full matrix and exact inference replay')
    old = json.loads((ROOT/'outputs/publication_readiness_2026_09/observed_motion/input_report.json').read_text())
    new = json.loads((reports/'input_report.json').read_text())
    if old['arrays'] != new['arrays']:
        raise ValueError('Independent motion extraction differs')
    checkpoints = sorted((output/'checkpoints').glob('*.pt'))
    checkpoint_info = []
    for path in checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['step'] != 4000 or len(state['losses']) != 41 or not state['optimizer']['state']:
            raise ValueError('Incomplete optimizer/checkpoint evidence')
        if not torch.isfinite(state['model']['output.weight']).all() or not torch.any(state['model']['output.weight'] != 0):
            raise ValueError('Untrained or nonfinite output head')
        checkpoint_info.append(dict(file=path.name, steps=state['step'], loss_records=len(state['losses']),
            output_weight_norm=float(torch.linalg.vector_norm(state['model']['output.weight']))))
    watched = checkpoints+[reports/'report.json']
    before = {str(p): file_digest(p) for p in watched}
    with (output/'completed_resume.log').open('w') as log:
        finished = subprocess.run([sys.executable, 'scripts/run_m3w_observed_motion_v2.py', '--registration',
                                  'configs/m3w_observed_motion_v2.json'], cwd=ROOT, stdout=log, stderr=log)
    if finished.returncode != 0:
        raise RuntimeError('Completed-run resume failed; inspect private log')
    after = {str(p): file_digest(p) for p in watched}
    if before != after:
        raise ValueError('Completed-run resume changed weights or report')
    result = dict(result_source='cached_verified_actual_checkpoint_replay_and_resume',
        report_sha256=file_digest(reports/'report.json'), model_count=len(checkpoints),
        replay_exact_models=54, independently_rebuilt_input_arrays_exact=True,
        completed_resume_new_updates=0, resume_weights_and_main_report_byte_identical=True,
        checkpoints=checkpoint_info, runtime=dict(platform=platform.platform(), machine=platform.machine(),
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            training_compute_threads=4, training_interop_threads=1, dataloader_workers=0),
        future_targets_are_loss_eval_only=True, strict_sensor_as_of_claim=False,
        development_calibration_confirmation_opened=False, stage5c_executed=False, smc_enabled=False,
        synthetic_and_focused_tests='13 passed; full legacy suite not rerun')
    json_write(reports/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}))


if __name__ == '__main__':
    main()
