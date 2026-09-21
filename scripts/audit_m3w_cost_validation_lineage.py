"""Verify whether frozen OOF producers support held-fold cost-head validation.

Read metadata and OOF row identities only. Hash checkpoints without deserializing
them; do not import Torch, evaluate predictions or access target array members.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.m3w_cost_validation_lineage import (
    audit_cost_validation, audit_prediction_lineage,
)
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_frozen_runtime import RelocatedCodeContract, SUPERVISED, MIRROR_FILES

CONFIG = 'configs/m3w_cost_validation_lineage_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def collect(root, cfg):
    root = Path(root).resolve()
    bindings, jobs, all_identities = {}, [], []

    def checked(relative, expected=None):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Input path escapes workspace')
        actual = file_digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Changed input: '+str(relative))
        if str(relative) in bindings and bindings[str(relative)] != actual:
            raise ValueError('Input changed during audit: '+str(relative))
        bindings[str(relative)] = actual
        return path

    protocol = json.loads(checked(cfg['protocol'], cfg['protocol_file_sha256']).read_text())
    for family, spec in cfg['families'].items():
        identity = json.loads(checked(f"{cfg['parent_private']}/{family}/identity.json",
                                      spec['identity_sha256']).read_text())
        report = json.loads(checked(f"{cfg['parent_public']}/{family}.json",
                                    spec['report_sha256']).read_text())
        if (report['run_sha256'] != digest(identity) or
                report['source_bindings'] != identity['source_bindings'] or identity['family'] != family):
            raise ValueError('Parent report/identity mismatch')
        for path, expected in identity['source_bindings'].items():
            checked(path, expected)
        runtime = identity['frozen_runtime']
        if (runtime['hash_checks_relaxed'] or runtime['original_checkout_modified'] or
                set(runtime['files']) != set(MIRROR_FILES)):
            raise ValueError('Invalid historical runtime declaration')
        mirror = Path(runtime['relocation'][SUPERVISED]).parents[2]
        for path, expected in runtime['files'].items():
            checked(str(mirror / path), expected)
        study = f'data/stage_cvpr2027_experiments/8to12_{family}_v6'
        for seed in cfg['seeds']:
            artifacts = [json.loads(checked(f'{study}/seed{seed}_{name}/artifact.json').read_text())
                         for name in ('full', 'hold0', 'hold1', 'hold2', 'ridge', 'neural_cost')]
            contract = RelocatedCodeContract(protocol, root, artifacts, runtime=runtime)
            if contract.digest != identity['protocol_sha256']:
                raise ValueError('Parent protocol mismatch')
            cache = f'{study}/seed{seed}_ridge'
            run = json.loads(checked(f'{cache}/run_identity.json').read_text())
            if run['protocol_sha256'] != contract.digest:
                raise ValueError('OOF cache protocol mismatch')
            groups, seen = {}, set()
            for fold in sorted(set(protocol['fit_folds'].values())):
                receipt = json.loads(checked(f'{cache}/fold_{fold}.json').read_text())
                source = checked(f'{cache}/fold_{fold}.npz', receipt['cache_sha256'])
                meta = receipt['group_metadata']
                producer = run['fold_models'][str(fold)]
                if (receipt['run_identity'] != run or meta['predictor_id'] != producer or
                        meta['protocol_sha256'] != contract.digest or
                        meta['predictor_sha256'] != contract.artifacts[producer]['sha256'] or
                        run['predictor_sha256'][producer] != meta['predictor_sha256'] or
                        meta['feature_source'] != 'past_and_frozen_rollouts_only'):
                    raise ValueError('OOF producer/receipt mismatch')
                with np.load(source, allow_pickle=False) as arrays:
                    rows = json.loads(arrays['identities_json'].item())
                records = sorted(r for r, f in protocol['fit_folds'].items() if f == fold)
                if not rows or sorted({r['recording_id'] for r in rows}) != records:
                    raise ValueError('Incomplete OOF recording identity')
                for row in rows:
                    name = row['recording_id']
                    if (row['data_role'] != 'fit' or row['protocol'] != 'observation_steps' or
                            row['physical_scene'] != protocol['records'][name]['physical_scene'] or
                            any(type(row[k]) is not int for k in ('agent_id', 'frame_id', 'horizon_raw'))):
                        raise ValueError('Invalid OOF row identity')
                    key = tuple(row[k] for k in ('recording_id', 'agent_id', 'frame_id', 'horizon_raw'))
                    if key in seen:
                        raise ValueError('Duplicate OOF row identity')
                    seen.add(key)
                original = audit_prediction_lineage(contract, producer, records)
                if not original['eligible_under_declared_lineage']:
                    raise ValueError('Original producer OOF guarantee failed')
                groups[fold] = dict(recordings=records, predictor_id=producer, rows=len(rows),
                                    rows_by_recording=dict(sorted(Counter(r['recording_id'] for r in rows).items())),
                                    original_oof_check=original)
            all_identities.append(seen)
            fitted = []
            for name in ('ridge', 'neural_cost'):
                head = f'seed{seed}_{name}'
                fitted.append(dict(head=head, folds=[audit_prediction_lineage(contract, head, g['recordings'])
                                                      for g in groups.values()]))
            outer_cases = []
            pool = [a['id'] for a in artifacts if a['kind'] == 'forecaster']
            for outer, group in groups.items():
                others = [g for f, g in groups.items() if f != outer]
                attempt = audit_cost_validation(contract, group['recordings'], others,
                                                validation_producer_id=group['predictor_id'])
                inner_cases = []
                for inner, other in groups.items():
                    if inner == outer:
                        continue
                    required = sorted(group['recordings'] + other['recordings'])
                    checks = [audit_prediction_lineage(contract, a, required) for a in pool]
                    inner_cases.append(dict(inner_fold=inner, required_excluded_recordings=required,
                        eligible_producers=[c['artifact_id'] for c in checks if c['eligible_under_declared_lineage']],
                        rejected=[dict(artifact_id=c['artifact_id'], exposed_scenes=sorted({s for e in c['exposures']
                                   for s in e['physical_scenes']})) for c in checks if not c['eligible_under_declared_lineage']]))
                outer_cases.append(dict(outer_fold=outer, reuse_other_oof_rows=attempt,
                                        existing_pool_inner_producer_search=inner_cases))
            jobs.append(dict(family=family, seed=seed, unique_rows=len(seen), folds=groups,
                             fitted_head_checks=fitted, hypothetical_outer_checks=outer_cases))
            print(json.dumps(dict(status='verified', family=family, seed=seed, rows=len(seen))), flush=True)
    if any(ids != all_identities[0] for ids in all_identities[1:]):
        raise ValueError('Family/seed cohorts differ')
    # Recheck inputs before publishing an audit bound to their bytes.
    for path, expected in list(bindings.items()):
        checked(path, expected)
    heads = [h for j in jobs for h in j['fitted_head_checks']]
    cases = [c for j in jobs for c in j['hypothetical_outer_checks']]
    inner = [i for c in cases for i in c['existing_pool_inner_producer_search']]
    return dict(result_source='fresh_run_lineage_audit_of_cached_verified_artifacts',
        protocol_sha256=contract.digest, source_bindings=bindings, jobs=jobs,
        summary=dict(families=len(cfg['families']), seeds=len(cfg['seeds']),
            unique_row_identities=len(all_identities[0]), unique_fit_scenes=len(contract.scene_set(protocol['fit_folds'])),
            original_oof_producer_checks=sum(len(j['folds']) for j in jobs),
            fitted_heads=len(heads), fitted_head_fold_checks=sum(len(h['folds']) for h in heads),
            fitted_head_fold_checks_rejected=sum(not f['eligible_under_declared_lineage'] for h in heads for f in h['folds']),
            hypothetical_outer_cases=len(cases),
            hypothetical_outer_cases_rejected=sum(not c['reuse_other_oof_rows']['eligible_under_declared_lineage'] for c in cases),
            outer_validation_producers_reusable=sum(c['reuse_other_oof_rows']['validation_producer']['eligible_under_declared_lineage'] for c in cases),
            required_outer_inner_cases=len(inner),
            cases_with_existing_eligible_inner_producer=sum(bool(i['eligible_producers']) for i in inner)),
        new_training=False, new_inference=False, checkpoint_deserialization=False,
        target_array_members_read=False, new_validation_split_adopted=False,
        primary_metric_changed=False, independent_calibration=False,
        independence_proven_beyond_declared_lineage=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    began = time.monotonic()
    cfg = json.loads((ROOT / CONFIG).read_text())
    out = ROOT / cfg['output'] / 'analysis.json'
    if out.exists() and not args.resume:
        raise ValueError('Audit already exists; use --resume for exact verification')
    result = collect(ROOT, cfg)
    result['code_bindings'] = {p: file_digest(ROOT / p) for p in (CONFIG,
        'scripts/audit_m3w_cost_validation_lineage.py', 'src/evaluation/m3w_cost_validation_lineage.py',
        'src/evaluation/m3w_experiment_contract.py', 'src/evaluation/m3w_frozen_runtime.py')}
    # Round-trip integer fold keys to the on-disk JSON representation.
    result = json.loads(json.dumps(result, allow_nan=False))
    if args.resume:
        if json.loads(out.read_text()) != result:
            raise ValueError('Completed lineage audit changed')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        os.replace(temp, out)
    print(json.dumps(dict(status='exact_replay' if args.resume else 'complete', pid=os.getpid(),
        elapsed_seconds=time.monotonic()-began, summary=result['summary'],
        report_sha256=file_digest(out), torch_imported='torch' in sys.modules)), flush=True)


if __name__ == '__main__':
    main()
