"""Read-only completed-cost verification anchored to earlier frozen fit reports.

Never backfill a completion receipt in an old training directory. The published
parent snapshots, rather than a newly hashed target cache, anchor old inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'

from src.evaluation.m3w_cost_completion import verify_completed_ridge
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_frozen_runtime import RelocatedCodeContract, SUPERVISED, MIRROR_FILES

PARENT_CONFIG = 'configs/m3w_cost_validation_lineage_v1.json'
OUTPUT = 'outputs/publication_readiness_2026_09/cost_completion_v2/analysis.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    began = time.monotonic()
    out = ROOT / OUTPUT
    if out.exists() and not args.resume:
        raise ValueError('Completed verification exists; use --resume')
    cfg = json.loads((ROOT / PARENT_CONFIG).read_text())
    bindings, results = {}, []

    def check(relative, expected):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or file_digest(path) != expected:
            raise ValueError('Trusted source binding changed: '+relative)
        if relative in bindings and bindings[relative] != expected:
            raise ValueError('Conflicting trusted source binding')
        bindings[relative] = expected
        return path

    protocol = json.loads(check(cfg['protocol'], cfg['protocol_file_sha256']).read_text())
    for family, spec in cfg['families'].items():
        identity = json.loads(check(f"{cfg['parent_private']}/{family}/identity.json", spec['identity_sha256']).read_text())
        parent = json.loads(check(f"{cfg['parent_public']}/{family}.json", spec['report_sha256']).read_text())
        if (parent['run_sha256'] != digest(identity) or
                parent['source_bindings'] != identity['source_bindings'] or identity['family'] != family):
            raise ValueError('Parent fit evidence differs from identity')
        for relative, sha in identity['source_bindings'].items():
            check(relative, sha)
        runtime = identity['frozen_runtime']
        if (runtime['hash_checks_relaxed'] or runtime['original_checkout_modified'] or
                set(runtime['files']) != set(MIRROR_FILES)):
            raise ValueError('Invalid historical runtime')
        mirror = Path(runtime['relocation'][SUPERVISED]).parents[2]
        for relative, sha in runtime['files'].items():
            check(str(mirror / relative), sha)
        sources = dict(code_sha256=ROOT / runtime['relocation'][SUPERVISED],
            oof_identity_source_sha256=ROOT / 'src/world_model/m3w_oof_identity.py',
            script_sha256=ROOT / 'scripts/train_m3w_oof_cost_head.py')
        study = ROOT / f'data/stage_cvpr2027_experiments/8to12_{family}_v6'
        for seed in cfg['seeds']:
            artifacts = [json.loads((study / f'seed{seed}_{name}/artifact.json').read_text())
                         for name in ('full', 'hold0', 'hold1', 'hold2')]
            contract = RelocatedCodeContract(protocol, ROOT, artifacts, runtime=runtime)
            result = verify_completed_ridge(contract, study / f'seed{seed}_ridge', source_paths=sources,
                                            expected_files=identity['source_bindings'])
            results.append(dict(family=family, seed=seed, **result))
            print(json.dumps(dict(family=family, seed=seed, status='dependencies_verified',
                                  training_rows=result['training_rows'])), flush=True)
    for relative, sha in list(bindings.items()):
        check(relative, sha)
    result = dict(result_source='fresh_run_integrity_verification_of_cached_verified_fit_assets',
        heads=results, source_bindings=bindings,
        implementation_bindings={p: file_digest(ROOT / p) for p in (
            PARENT_CONFIG, 'scripts/verify_m3w_frozen_cost_completions.py',
            'scripts/train_m3w_oof_cost_head_v2.py', 'src/evaluation/m3w_cost_completion.py',
            'src/world_model/m3w_oof_identity.py', 'src/evaluation/m3w_experiment_contract.py',
            'src/evaluation/m3w_frozen_runtime.py')},
        summary=dict(heads=len(results), normalization_checks_passed=sum(r['normalization_exact'] for r in results),
                     all_targets_externally_bound=all(r['target_source_status']=='externally_bound_bytes_verified' for r in results)),
        new_training=False, new_inference=False, old_training_directories_modified=False,
        primary_metric_changed=False, independent_calibration=False, deployment_approved=False,
        stage5c_executed=False, smc_enabled=False)
    if args.resume:
        if json.loads(out.read_text()) != result:
            raise ValueError('Completed verification changed')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, out)
    print(json.dumps(dict(status='exact_replay' if args.resume else 'complete', summary=result['summary'],
        pid=os.getpid(), elapsed_seconds=time.monotonic()-began, torch_imported='torch' in sys.modules,
        report_sha256=file_digest(out))), flush=True)


if __name__ == '__main__':
    main()
