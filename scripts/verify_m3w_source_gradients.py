"""Persist exact replay and immutable-checkpoint evidence for gradient audit."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import immutable_json
from scripts.verify_m3w_source_motion_quality import preserve_verification
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_gradient_diagnostic import gradient_relation
import numpy as np


def main():
    registration = Path('configs/m3w_source_gradient_diagnostic_v1.json')
    reg = json.loads(registration.read_text())
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    audit = json.loads((public/'audit.json').read_text())
    hashes = {}
    for trial in audit['trials']:
        path = ROOT/trial['gradient_archive']
        assert file_digest(path) == trial['gradient_sha256']
        assert file_digest(ROOT/trial['checkpoint_path']) == trial['checkpoint_sha256']
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['static_gradient']+a['nonzero_gradient'], a['full_gradient'])
            assert gradient_relation(a['static_gradient'], a['full_gradient']) == trial['static_vs_full']
            for proposal in trial['proposals']:
                assert gradient_relation(a[proposal+'_raw_mean'], a['full_gradient']) == trial['proposals'][proposal]['raw_mean_vs_population']
                assert gradient_relation(a[proposal+'_clipped_mean'], a['full_gradient']) == trial['proposals'][proposal]['clipped_mean_vs_population']
        for path in [path, ROOT/trial['checkpoint_path'], private/'trials'/(trial['trial']+'.json')]:
            hashes[str(path.relative_to(ROOT))] = file_digest(path)
    for path in [public/'audit.json', private/'identity.json', ROOT/registration]:
        hashes[str(path.relative_to(ROOT))] = file_digest(path)
    command = [sys.executable, 'scripts/audit_m3w_source_gradients.py', '--registration', str(registration)]
    replay = subprocess.run(command+['--replay'], cwd=ROOT, capture_output=True, text=True, check=True)
    events = [json.loads(line) for line in replay.stdout.splitlines() if line.startswith('{')]
    exact = [e['trial'] for e in events if e['state'] == 'exact_replay']
    assert len(set(exact)) == 24 and events[-1]['fresh_probes'] == 24
    immutable_json(public/'replay.json', dict(exact_replays=exact, new_updates=0, new_held_predictions=0))
    resume = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    last = [json.loads(line) for line in resume.stdout.splitlines() if line.startswith('{')][-1]
    assert last['fresh_probes'] == 0 and last['cached_verified'] == 24
    assert hashes == {p: file_digest(ROOT/p) for p in hashes}
    result = dict(result_source='fresh_run_exact_gradient_replay_and_readonly_resume',
        exact_replays=24, gradient_partition_checks=24, probe_mean_vector_checks=96,
        unchanged_artifacts=len(hashes), artifact_hashes=hashes, completed_resume=last,
        new_updates=0, new_held_predictions=0, new_deployment=False)
    preserve_verification(public/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
