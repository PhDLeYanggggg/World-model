"""Verify lightweight OnDemand downloads without inventing scheduler acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from scripts import read_m3w_easy_harm_final_heads_v2 as heads
from scripts.diagnose_m3w_easy_harm_node_portability import summarize

HOME = heads.manager.PUBLIC
ROOT = heads.ROOT
COLLECTION = HOME / 'web_collection_v1'
JOIN_OUT = 'recovery-v2-join-37835859_4294967294.out'
JOIN_ERR = 'recovery-v2-join-37835859_4294967294.err'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_files(path, expected):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        files = [entry for entry in entries if not entry.is_dir()]
        if (len(files) != len(expected) or len(entries) > len(expected) + 1
                or {entry.filename for entry in files} != set(expected)
                or any(entry.is_dir() and entry.filename != './' for entry in entries)
                or sum(entry.file_size for entry in entries) > 8 * 2**20):
            raise ValueError('Exact bounded archive inventory required')
        for entry in files:
            name = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (name.name != entry.filename or name.is_absolute()
                    or entry.file_size > 2 * 2**20 or entry.flag_bits & 1
                    or stat.S_ISLNK(mode)):
                raise ValueError('Only regular top-level receipt files allowed')
        return {entry.filename: archive.read(entry) for entry in files}


def verify_bindings():
    names = ('registration.json', 'control_execution_amendment.json',
             'control_execution_amendment_v2.json', 'readout/calculation_registration.json',
             'readout/registration.json', 'readout_verified_v2/registration.json')
    checked = {}
    for name in names:
        record = json.loads((HOME / name).read_text())
        for rel, expected in record['bindings'].items():
            path = (ROOT / rel).resolve()
            if not path.is_relative_to(ROOT.resolve()) or digest(path.read_bytes()) != expected:
                raise ValueError('Registered source changed: ' + rel)
        checked[name] = len(record['bindings'])
    return checked


def verify_downloads(downloads):
    bindings = verify_bindings()
    freeze_raw = (downloads / 'training_freeze.json').read_bytes()
    freeze = json.loads(freeze_raw)
    reg = json.loads(heads.manager.REG.read_text())
    cfg = json.loads(heads.manager.CONFIG.read_text())
    amendment = json.loads((HOME / heads.recovery.AMENDMENT).read_text())
    wanted = {PurePosixPath(ref['path']).name: ref['path'] for ref in freeze['fits']}
    contents = archive_files(downloads / 'fits.zip', wanted)
    files = {wanted[name]: raw for name, raw in contents.items()}
    docs = heads.validate(freeze, files, reg['expected'], cfg,
                          heads.manager.base.sha(heads.manager.REG), amendment,
                          heads.manager.base.sha(HOME / heads.recovery.AMENDMENT))
    submission_raw = (downloads / 'join_submission.json').read_bytes()
    submission = json.loads(submission_raw)
    if (submission['job_id'] != '37835859'
            or submission['registration_sha256'] != freeze['registration_sha256']
            or submission['execution_amendment_sha256'] != freeze['control_execution_amendment_sha256']):
        raise ValueError('Owned join submission required')
    join_raw = (downloads / JOIN_OUT).read_bytes()
    error_raw = (downloads / JOIN_ERR).read_bytes()
    if json.loads(join_raw) != dict(state='all144_frozen',
                                   counts=dict(historical=54, replay=18, candidate=72)) or error_raw:
        raise ValueError('Complete join output and empty error log required')
    outputs = {ROOT / rel: raw for rel, raw in files.items()}
    outputs.update({HOME / 'training_freeze.json': freeze_raw,
                    COLLECTION / 'join_submission.json': submission_raw,
                    COLLECTION / 'join_stdout.txt': join_raw,
                    COLLECTION / 'join_stderr.txt': error_raw})
    receipt = dict(result_source='fresh_run_web_collection_of_cached_verified_training_receipts',
        transport='official_CREATE_Open_OnDemand_browser_download',
        remote_root=heads.manager.REMOTE, source_registrations_checked=bindings,
        fit_receipts=len(docs), preserved_accepted_fit_hashes=110,
        historical_exact_controls=54, exact_original_replay_controls=18,
        historical_bitwise_reproduction_complete=False, candidate_fits=72,
        historical_recovery_fresh_candidate_fits=17, fresh_training_this_collection=False,
        join_artifact_complete=True, join_accounting_status='unknown_not_retrieved',
        scheduler_acceptance=False, readout_admission=False, checkpoint_bytes_downloaded=0,
        checkpoint_binaries_rehashed_this_collection=False, scientific_readout='not_run',
        independent_roles_read=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False,
        downloads={name: digest((downloads / name).read_bytes()) for name in
                   ('training_freeze.json', 'fits.zip', 'join_submission.json', JOIN_OUT, JOIN_ERR)})
    return outputs, receipt


def verify_portability(downloads, outputs, receipt):
    path = downloads / 'node_portability_v1.zip'
    with zipfile.ZipFile(path) as archive:
        complete_raw = archive.read('complete.json')
    complete = json.loads(complete_raw)
    reg_path = HOME / 'node_portability_v1_registration.json'
    registration = json.loads(reg_path.read_text())
    if (complete['registration_sha256'] != digest(reg_path.read_bytes())
            or complete['job_id'] != '37837636' or complete['node'] != registration['node']):
        raise ValueError('Owned registered portability diagnostic required')
    refs = complete['results']
    expected = {PurePosixPath(ref['path']).name for ref in refs} | {'complete.json'}
    if len(refs) != 17 or len(expected) != 18:
        raise ValueError('Exactly17 distinct diagnostic records required')
    contents = archive_files(path, expected)
    amendment = json.loads((HOME / heads.recovery.AMENDMENT).read_text())
    rows = []
    for ref in refs:
        name = PurePosixPath(ref['path']).name
        target = HOME / 'node_portability_v1' / name
        if str(target.relative_to(ROOT)) != ref['path'] or digest(contents[name]) != ref['sha256']:
            raise ValueError('Diagnostic path or hash mismatch')
        row = json.loads(contents[name])
        if (row['registration_sha256'] != complete['registration_sha256']
                or row['job_id'] != complete['job_id']
                or row['reference_checkpoint'] != amendment['references'][row['name']]['reference_checkpoint']
                or row['validation_scored'] is not False or row['independent_roles_read'] is not False
                or row['verification_updates'] != 2000 or row['new_scientific_fits'] != 0):
            raise ValueError('Diagnostic provenance changed')
        rows.append(row)
        outputs[target] = contents[name]
    summary = summarize(rows, registration['identities'])
    if summary != complete['summary']:
        raise ValueError('Diagnostic summary mismatch')
    outputs[HOME / 'node_portability_v1' / 'complete.json'] = complete_raw
    receipt['portability'] = dict(summary=summary, job_id=complete['job_id'],
        scheduler_accounting='unknown_not_retrieved', archive_sha256=digest(path.read_bytes()))


def install(outputs):
    # Check every existing destination before writing any new receipt.
    for path, raw in outputs.items():
        if path.is_symlink() or (path.exists() and path.read_bytes() != raw):
            raise ValueError('Refuse to replace existing evidence: ' + str(path))
    for path, raw in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    outputs, receipt = verify_downloads(args.downloads)
    verify_portability(args.downloads, outputs, receipt)
    receipt['outputs'] = {str(path.relative_to(ROOT)): dict(sha256=digest(raw), bytes=len(raw))
                          for path, raw in sorted(outputs.items())}
    receipt_path = COLLECTION / 'receipt.json'
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if {key: value for key, value in existing.items() if key != 'collected_utc'} != receipt:
            raise ValueError('Collection changed; preserve original receipt')
        receipt = existing
    else:
        receipt['collected_utc'] = datetime.now(timezone.utc).isoformat()
    if args.apply:
        outputs[receipt_path] = (json.dumps(receipt, indent=2) + '\n').encode()
        install(outputs)
    print(json.dumps(dict(applied=args.apply, fit_receipts=receipt['fit_receipts'],
        portability=receipt['portability']['summary'],
        receipt=str(receipt_path.relative_to(ROOT)) if args.apply else None,
        join_accounting_status=receipt['join_accounting_status'],
        scheduler_acceptance=False, readout_admission=False, scientific_readout='not_run'), indent=2))


if __name__ == '__main__':
    main()
