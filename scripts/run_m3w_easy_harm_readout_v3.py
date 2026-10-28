"""Revalidate immutable payloads; preserve unknown historical scheduler status."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import platform
from pathlib import Path
import shlex
import subprocess

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')

from scripts import run_m3w_easy_harm_readout_v2 as previous
from scripts import import_m3w_easy_harm_web_artifacts as importer
from scripts import verify_m3w_easy_harm_artifacts_v3 as worker

heads, original = previous.heads, previous.original
ROOT, HOME = heads.ROOT, heads.manager.PUBLIC
PUBLIC = HOME / 'readout_verified_v3'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/readout_verified_v3'
WORKER = ROOT / 'scripts/verify_m3w_easy_harm_artifacts_v3.py'
REMOTE = heads.manager.REMOTE + '/' + worker.DIRECTORY
JOB_NAME = 'm3w_frozen_artifact_v3'
sha, once = original.sha, original.once


def frozen_documents():
    importer.verify_bindings()
    reg = json.loads(heads.manager.REG.read_text())
    cfg = json.loads(heads.manager.CONFIG.read_text())
    amend = json.loads((HOME / heads.recovery.AMENDMENT).read_text())
    if (sha(heads.manager.CONFIG) != reg['config_sha256']
            or amend['training_registration_sha256'] != sha(heads.manager.REG)
            or amend['previous_amendment_sha256'] != sha(HOME / 'control_execution_amendment.json')):
        raise ValueError('Changed original training contract')
    for ref in amend['proofs']:
        if sha(ROOT / ref['path']) != ref['sha256']:
            raise ValueError('Changed exact-reference proof')
    freeze = json.loads((HOME / 'training_freeze.json').read_text())
    files = {}
    for ref in freeze['fits']:
        path = ROOT / ref['path']
        if path.is_symlink() or not path.resolve().is_relative_to((HOME / 'fits').resolve()):
            raise ValueError('Misplaced training receipt')
        files[ref['path']] = path.read_bytes()
    docs = heads.validate(freeze, files, reg['expected'], cfg, sha(heads.manager.REG),
                          amend, sha(HOME / heads.recovery.AMENDMENT))
    return docs, dict(training_freeze_sha256=sha(HOME / 'training_freeze.json'),
                     checkpoints={r['checkpoint']['path']: r['checkpoint'] for r in docs.values()})


def registration():
    cfg, old = previous.registration()
    if old != json.loads((previous.PUBLIC / 'registration.json').read_text()):
        raise ValueError('Prior readout calculation or admission changed')
    docs, inventory = frozen_documents()
    files = [dict(path=str((HOME / 'training_freeze.json').relative_to(ROOT)),
                  sha256=inventory['training_freeze_sha256'],
                  bytes=(HOME / 'training_freeze.json').stat().st_size, kind='freeze')]
    freeze = json.loads((HOME / 'training_freeze.json').read_text())
    files += [dict(**r, bytes=(ROOT / r['path']).stat().st_size, kind='receipt') for r in freeze['fits']]
    files += [dict(**r['checkpoint'], kind='checkpoint') for r in docs.values()]
    manifest = dict(files=sorted(files, key=lambda r:r['path']), checkpoint_count=144,
        worker_sha256=sha(WORKER), training_registration_sha256=sha(heads.manager.REG),
        amendment_sha256=sha(HOME / heads.recovery.AMENDMENT))
    paths = [Path(__file__).resolve(), WORKER, ROOT / 'tests/test_m3w_artifact_readout_v3.py',
             PUBLIC / 'admission_protocol.md']
    reg = dict(previous_registration_sha256=sha(previous.PUBLIC / 'registration.json'),
        bindings={str(p.relative_to(ROOT)):sha(p) for p in paths}, manifest=manifest,
        scientific_calculation_unchanged=True, historical_accounting='unavailable_not_inferred',
        new_admission='fresh_allocated_full_payload_verification_and_successful_controller_exit',
        local_checkpoint_cache=False, independent_roles_read=False, new_training=False,
        threshold_search=False, checkpoint_selection=False, deployment_changed=False,
        risk_budget=.02, stage5c_executed=False, smc_enabled=False)
    return cfg, reg


def controller_success(record, job):
    if record.get('returncode') != 0:
        return False
    fields = dict(x.split('=', 1) for x in shlex.split(record['stdout']) if '=' in x)
    return (fields.get('JobId') == job and fields.get('JobName') == JOB_NAME
            and fields.get('JobState') == 'COMPLETED' and fields.get('ExitCode') == '0:0'
            and fields.get('WorkDir') == REMOTE)


def check_report(report, reg, job):
    manifest = reg['manifest']
    expected = {r['path']:{k:r[k] for k in ('sha256','bytes','kind')} for r in manifest['files']}
    if (report['schema'] != 'm3w_frozen_artifact_verification_v3' or report['job_id'] != job
            or report['worker_sha256'] != manifest['worker_sha256']
            or report['manifest_sha256'] != sha(PUBLIC / 'manifest.json')
            or report['files'] != expected or report['bytes_verified'] != sum(r['bytes'] for r in expected.values())
            or report['original_training_bindings_verified'] is not True
            or report['historical_scheduler_status'] != 'unavailable_not_inferred'
            or any(report[k] is not False for k in ('scientific_execution','model_deserialization',
                'new_training','existing_artifacts_changed','independent_roles_read','stage5c_executed','smc_enabled'))):
        raise ValueError('Complete unchanged allocated verification required')


SUBMIT = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); payload=json.loads(sys.stdin.buffer.read())
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
manifest=json.loads(payload['manifest.json'])
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==manifest['training_registration_sha256']
assert hashlib.sha256(payload['worker.py'].encode()).hexdigest()==manifest['worker_sha256']
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and not any('|m3w_frozen_artifact_v3|' in x for x in q.stdout.splitlines())
work=root/'readout_artifact_v3';work.mkdir(exist_ok=False)
for name,raw in payload.items():
 assert name in ('manifest.json','worker.py','verify.sbatch')
 with (work/name).open('x') as f:f.write(raw)
with (work/'submission_intent.json').open('x') as f:json.dump(dict(scientific_execution=False,existing_jobs_changed=False),f)
try:
 p=subprocess.run(['sbatch','--parsable',str(work/'verify.sbatch')],capture_output=True,text=True,timeout=30)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_inspect_do_not_resubmit')
with (work/'submission.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''


def submit(reg):
    if (PUBLIC / 'submission.json').exists():
        raise RuntimeError('Existing submission; observe it instead of resubmitting')
    script = '\n'.join(['#!/bin/bash -l', '#SBATCH --job-name='+JOB_NAME,
        '#SBATCH --account=kcl', '#SBATCH --partition=cpu', '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=1', '#SBATCH --mem=1G',
        '#SBATCH --time=00:05:00', '#SBATCH --no-requeue', '#SBATCH --chdir='+REMOTE,
        '#SBATCH --output='+REMOTE+'/verify-%j.out', '#SBATCH --error='+REMOTE+'/verify-%j.err',
        'set -euo pipefail', 'exec /usr/bin/python3 -I -B '+REMOTE+'/worker.py', ''])
    payload = json.dumps({'manifest.json':(PUBLIC / 'manifest.json').read_text(),
                         'worker.py':WORKER.read_text(), 'verify.sbatch':script}).encode()
    result = heads.manager.base.remote(SUBMIT, [heads.manager.REMOTE], payload, timeout=50)
    once(PUBLIC / 'submission.json', result)
    print(json.dumps(result), flush=True)


def observe(reg):
    submission = json.loads((PUBLIC / 'submission.json').read_text())
    job = submission['job_id']
    code = r'''
import json,pathlib,subprocess,sys
work=pathlib.Path(sys.argv[1]);job=sys.argv[2]
assert work==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1/readout_artifact_v3')
assert json.loads((work/'submission.json').read_text())['job_id']==job and job.isdigit()
cmd=['scontrol','show','job',job,'-o'];p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
out=dict(controller=dict(argv=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),files={})
for name in ('report.json','verify-'+job+'.out','verify-'+job+'.err'):
 path=work/name
 if path.exists():
  assert path.stat().st_size<2*2**20
  out['files'][name]=path.read_text()
print(json.dumps(out))
'''
    result = heads.manager.base.remote(code, [REMOTE, job], timeout=35)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    once(PUBLIC / ('observation_'+stamp+'.json'), result)
    admitted = False
    if controller_success(result['controller'], job):
        report = json.loads(result['files']['report.json'])
        check_report(report, reg, job)
        if result['files']['verify-'+job+'.err']:
            raise ValueError('Verification stderr is not empty')
        stdout = json.loads(result['files']['verify-'+job+'.out'])
        import hashlib
        if (stdout['state'] != 'all144_payloads_verified' or stdout['job_id'] != job
                or stdout['report_sha256'] != hashlib.sha256(result['files']['report.json'].encode()).hexdigest()):
            raise ValueError('Verification output mismatch')
        once(PUBLIC / 'accepted.json', dict(job_id=job, controller=result['controller'], report=report,
             historical_scheduler_status='unavailable_not_inferred',
             result_source='fresh_run_full_payload_verification_of_cached_training',
             scientific_readout='not_run'))
        admitted = True
    print(json.dumps(dict(job_id=job, controller=result['controller'], admitted=admitted)), flush=True)


def admission(reg):
    docs, inventory = frozen_documents()
    accepted = json.loads((PUBLIC / 'accepted.json').read_text())
    job = json.loads((PUBLIC / 'submission.json').read_text())['job_id']
    if (accepted['job_id'] != job or not controller_success(accepted['controller'], job)
            or accepted['historical_scheduler_status'] != 'unavailable_not_inferred'):
        raise ValueError('Fresh successful verification allocation required')
    check_report(accepted['report'], reg, job)
    return docs, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register','submit','observe','preflight','run'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, reg = registration()
    if args.phase == 'register':
        once(PUBLIC / 'manifest.json', reg['manifest'])
        once(PUBLIC / 'registration.json', reg)
        print(json.dumps(dict(registered=True, checkpoint_count=144, scientific_readout='not_run')))
        return
    if reg != json.loads((PUBLIC / 'registration.json').read_text()):
        raise ValueError('Changed execution registration')
    for rel in [*reg['bindings'], str((PUBLIC / 'registration.json').relative_to(ROOT)),
                str((PUBLIC / 'manifest.json').relative_to(ROOT))]:
        if subprocess.check_output(['git','show','HEAD:'+rel], cwd=ROOT) != (ROOT / rel).read_bytes():
            raise ValueError('Commit frozen execution amendment before use')
    if args.phase == 'submit':
        submit(reg)
        return
    if args.phase == 'observe':
        observe(reg)
        return
    docs, inventory = admission(reg)
    if args.phase == 'preflight':
        print(json.dumps(dict(complete_heads=len(docs), payloads_reverified=True, scientific_readout='not_run')))
        return
    original.parent.base.inter.committed(PUBLIC / 'accepted.json')
    if (PUBLIC / 'complete.json').exists():
        raise RuntimeError('Completed readout is immutable')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    original.fitting.torch.set_num_threads(4)
    original.fitting.torch.set_num_interop_threads(1)
    original.PUBLIC, original.PRIVATE, original.heads = PUBLIC, PRIVATE, heads
    # The original evaluator, causal inputs and all frozen inference are unchanged.
    _, old_reg = previous.registration()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.run(cfg, old_reg, docs, inventory, args.resume)


if __name__ == '__main__':
    main()
