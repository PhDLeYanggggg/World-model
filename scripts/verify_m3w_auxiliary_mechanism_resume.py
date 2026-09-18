"""Replay a completed trainer invocation and verify immutable artifact hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads((ROOT/args.registration).read_text())
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    report_path = reports/'report.json'
    report = json.loads(report_path.read_text())
    if not report['complete'] or report['new_fits'] != 54:
        raise ValueError('Completed fixed matrix required')
    paths = sorted([*output.glob('checkpoints/*.pt'), *output.glob('predictions/*.npz'),
                    *output.glob('trials/*.json'), *output.glob('donors/*.npy'),
                    output/'training_identity.json'])
    if len(paths) != 166:
        raise ValueError('Expected 166 immutable run artifacts')
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report_hash = digest(report_path)
    command = [sys.executable, 'scripts/run_m3w_auxiliary_mechanism.py',
               '--registration', str(args.registration)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    final = events[-1]
    if final.get('state') != 'completed_resume_verified' or final.get('new_optimizer_updates') != 0:
        raise ValueError('Trainer did not verify a zero-update completed resume')
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if after != before or digest(report_path) != report_hash:
        raise ValueError('Completed resume changed immutable artifacts or aggregate results')
    combined = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    evidence = dict(result_source='fresh_run_completed_resume_verification',
        registration_sha256=digest(ROOT/args.registration), report_sha256=report_hash,
        immutable_artifacts=len(paths), all_hashes_unchanged=True,
        artifact_manifest_sha256=combined, artifacts=before,
        new_optimizer_updates=0, trainer_final_event=final,
        excluded_mutable_outputs=['training_heartbeat.json', 'input_checks.json'],
        independent_confirmation=False)
    (reports/'resume_verification.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'artifacts'}))


if __name__ == '__main__':
    main()
