"""Run the versioned real development study, sequentially and resumably."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    protocol = ROOT / 'configs/m3w_8to12_development_v1.json'
    contract = ExperimentContract(json.loads(protocol.read_text()), ROOT)
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or not set(args.seeds) <= set(contract.protocol['seeds']) or args.threads < 1):
        raise SystemExit('Explicit distinct protocol-listed seeds and positive threads required')
    output = ROOT / 'data/stage_cvpr2027_experiments/8to12_v1'
    output.mkdir(parents=True, exist_ok=True)
    common = ['--protocol', str(protocol), '--device', args.device, '--threads', str(args.threads)]
    baseline = 'constant_velocity_causal_fd'
    fit = sorted(contract.protocol['fit_folds'])
    folds = sorted(set(contract.protocol['fit_folds'].values()))

    def run(name, script, options):
        command = [sys.executable, str(ROOT / 'scripts' / script), *common, *options]
        started = time.monotonic()
        with (output / (name + '.log')).open('a') as log:
            log.write(json.dumps({'command': command}) + '\n')
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    write_json(output / 'runner_heartbeat.json', {
                        'runner_pid': os.getpid(), 'child_pid': child.pid, 'stage': name,
                        'elapsed_seconds': time.monotonic() - started, 'protocol_sha256': contract.digest,
                        'result_source': 'running_not_complete', 'command': command})
                    try:
                        code = child.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                child.send_signal(signal.SIGINT)
                child.wait()
                raise
        if code:
            raise SystemExit(f'{name} failed with exit {code}; inspect {output / (name + ".log")}')
        print(json.dumps({'completed': name, 'elapsed_seconds': time.monotonic() - started}), flush=True)

    for seed in args.seeds:
        artifacts, producers = [], {}
        specs = [('full', fit)] + [(f'hold{fold}', [r for r in fit if contract.protocol['fit_folds'][r] != fold])
                                   for fold in folds]
        for name, recordings in specs:
            directory = output / f'seed{seed}_{name}'
            options = ['--config', str(ROOT / 'configs/m3w_intervention_backend.json'),
                       '--fit-recordings', *recordings, '--baseline', baseline, '--seed', str(seed),
                       '--output-dir', str(directory)]
            if (directory / 'latest.pt').exists():
                options.append('--resume')
            run(directory.name, 'train_m3w_causal_forecaster.py', options)
            artifacts.append(directory / 'artifact.json')
            if name != 'full':
                producers[name.removeprefix('hold')] = directory.name
        mapping = output / f'seed{seed}_fold_models.json'
        if mapping.exists() and json.loads(mapping.read_text()) != producers:
            raise SystemExit('Existing fold map changed')
        write_json(mapping, producers)
        ridge = output / f'seed{seed}_ridge'
        options = ['--artifacts', *map(str, artifacts), '--fold-models', str(mapping), '--baseline', baseline,
                   '--alpha', '1', '--output-dir', str(ridge)]
        if ridge.exists():
            options.append('--resume')
        run(ridge.name, 'train_m3w_oof_cost_head.py', options)
        neural = output / f'seed{seed}_neural_cost'
        options = ['--artifacts', *map(str, artifacts), '--oof-cache-dir', str(ridge),
                   '--seed', str(seed), '--output-dir', str(neural)]
        if (neural / 'latest.pt').exists():
            options.append('--resume')
        run(neural.name, 'train_m3w_neural_cost_head.py', options)
        candidates = []
        for directory in (ridge, neural):
            report = directory / 'fit_report.json'
            artifacts.append(directory / 'artifact.json')
            for policy in contract.protocol['development_evaluation']['policies']:
                candidates.append({'id': f'{directory.name}_{policy}', 'policy_id': policy,
                    'forecaster_id': f'seed{seed}_full', 'risk_head_id': directory.name, 'baseline': baseline,
                    'risk_report_path': str(report.relative_to(ROOT)), 'risk_report_sha256': file_digest(report)})
        plan = {'schema_version': 1, 'candidates': candidates}
        plan_path = output / f'seed{seed}_development_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise SystemExit('Frozen development plan changed')
        write_json(plan_path, plan)
        evaluation = output / f'seed{seed}_development'
        options = ['--artifacts', *map(str, artifacts), '--plan', str(plan_path), '--output-dir', str(evaluation)]
        if evaluation.exists():
            options.append('--resume')
        run(evaluation.name, 'evaluate_m3w_development.py', options)
    write_json(output / 'runner_heartbeat.json', {'state': 'requested_seeds_complete', 'seeds': args.seeds,
               'protocol_sha256': contract.digest, 'independent_confirmation': False})


if __name__ == '__main__':
    main()
