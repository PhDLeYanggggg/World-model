import copy
import hashlib
import json

import pytest

from scripts import run_m3w_easy_harm_readout_v3 as run
from scripts import verify_m3w_easy_harm_artifacts_v3 as worker


def inventory(tmp_path):
    files = []
    for i in range(144):
        for kind, prefix, suffix in (
            ('checkpoint','data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/heads','.pt.gz'),
            ('receipt','outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/fits','.json')):
            path = tmp_path / prefix / (str(i)+suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(str(i).encode())
            files.append(dict(path=str(path.relative_to(tmp_path)),kind=kind,
                              bytes=path.stat().st_size,sha256=worker.sha(path)))
    path = tmp_path / 'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/training_freeze.json'
    path.write_bytes(b'{}')
    files.append(dict(path=str(path.relative_to(tmp_path)),kind='freeze',bytes=2,sha256=worker.sha(path)))
    return dict(files=files,checkpoint_count=144)


def test_complete_payload_hash_verification(tmp_path):
    manifest = inventory(tmp_path)
    checked, total = worker.verify_files(tmp_path, manifest)
    assert len(checked) == 289
    assert total == sum(r['bytes'] for r in manifest['files'])


@pytest.mark.parametrize('change',['missing','duplicate','hash','size','escape','symlink','kind'])
def test_payload_changes_fail_closed(tmp_path, change):
    manifest = inventory(tmp_path)
    row = manifest['files'][0]
    if change == 'missing': manifest['files'].pop()
    elif change == 'duplicate': manifest['files'][2] = row.copy()
    elif change == 'hash': row['sha256'] = '0'*64
    elif change == 'size': row['bytes'] += 1
    elif change == 'escape': row['path'] = '../outside'
    elif change == 'kind': row['kind'] = 'receipt'
    else:
        path = tmp_path / row['path']
        path.unlink()
        path.symlink_to(tmp_path / manifest['files'][2]['path'])
    with pytest.raises(ValueError): worker.verify_files(tmp_path, manifest)


def controller(**changes):
    fields = dict(JobId='123',JobName=run.JOB_NAME,JobState='COMPLETED',ExitCode='0:0',WorkDir=run.REMOTE)
    fields.update(changes)
    return dict(returncode=0,stdout=' '.join(k+'='+v for k,v in fields.items()))


def test_only_exact_new_successful_controller_job_is_accepted():
    assert run.controller_success(controller(), '123')
    for fields in [dict(JobState='RUNNING'),dict(ExitCode='1:0'),dict(JobId='124'),
                   dict(JobName='other'),dict(WorkDir='/other')]:
        assert not run.controller_success(controller(**fields), '123')
    assert not run.controller_success(dict(returncode=1,stdout=controller()['stdout']), '123')
    assert not run.controller_success(dict(returncode=0,stdout=''), '123')


def test_readout_still_calls_original_calculation_and_stream_reader():
    source = (run.ROOT / 'scripts/run_m3w_easy_harm_readout_v3.py').read_text()
    assert 'original.run(cfg, old_reg, docs, inventory, args.resume)' in source
    assert 'original.heads = PUBLIC, PRIVATE, heads' in source
    assert 'train_collection.json' not in source
    assert run.heads.Reader is run.previous.heads.Reader


def test_current_frozen_metadata_and_all_original_bindings_verify():
    docs, inv = run.frozen_documents()
    assert len(docs) == len(inv['checkpoints']) == 144
    assert inv['training_freeze_sha256'] == 'f57acbc74ba9afa47550b7c5aa37e70a42c2a14bd777ac9d4e9d8b2ee9542426'


def test_report_cannot_replace_manifest_or_claim_science(tmp_path, monkeypatch):
    manifest = inventory(tmp_path)
    manifest['worker_sha256'] = 'a'*64
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    files, total = worker.verify_files(tmp_path, manifest)
    report = dict(schema='m3w_frozen_artifact_verification_v3',job_id='123',
        worker_sha256='a'*64,manifest_sha256=run.sha(tmp_path/'manifest.json'),
        files=files,bytes_verified=total,original_training_bindings_verified=True,
        historical_scheduler_status='unavailable_not_inferred',scientific_execution=False,
        model_deserialization=False,new_training=False,existing_artifacts_changed=False,
        independent_roles_read=False,stage5c_executed=False,smc_enabled=False)
    run.check_report(report,dict(manifest=manifest),'123')
    for key, value in [('files',{}),('job_id','124'),('scientific_execution',True),
                       ('historical_scheduler_status','COMPLETED')]:
        wrong = copy.deepcopy(report)
        wrong[key] = value
        with pytest.raises(ValueError): run.check_report(wrong,dict(manifest=manifest),'123')
