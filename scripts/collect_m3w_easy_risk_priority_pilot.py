"""Verify the completed real fitting pilot before continuing its checkpoints."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, REMOTE, CONFIG, registration, digest, immutable
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote
from scripts.report_m3w_easy_risk_priority_training import remote_config_digest


def main():
    assert registration() == json.loads((PUBLIC/'registration.json').read_text())
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home'])
assert json.loads((root/'.owner.json').read_text())=={'project':'M3W','experiment':'european_easy_risk_priority_v1'}
doc=json.loads((root/'pilot.json').read_text())
assert doc['job_id']==json.loads((root/'submit_receipt_pilot.json').read_text())['job_id']
a=subprocess.run(['sacct','-j',doc['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
assert doc['registration_sha256']==p['registration_sha256'] and doc['config_sha256']==p['config_sha256']
assert doc['groups']==1 and doc['heads']==2 and len(doc['artifacts'])==2 and doc['updates_per_head']==100
assert not doc['held_outcomes_used'] and not doc['independent_roles_read']
size=0
for ref in doc['artifacts']:
    rel=pathlib.PurePosixPath(ref['path'])
    assert rel==pathlib.PurePosixPath('heads')/ref['group']/ref['arm']/'checkpoint.pt.gz'
    assert not rel.is_absolute() and '..' not in rel.parts and ref['arm'] in ('uncapped','risk_priority')
    data=(root/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256'];size+=len(data)
print(json.dumps(dict(receipt=doc,receipt_sha256=hashlib.sha256((root/'pilot.json').read_bytes()).hexdigest(),checkpoint_bytes=size,hash_checked_files=2)))
'''
    r = call_remote(code, dict(home=REMOTE, registration_sha256=digest(PUBLIC/'registration.json'), config_sha256=remote_config_digest(CONFIG)))
    if r['returncode'] != 0:
        raise RuntimeError(r.get('stderr', 'Observation unavailable')[-1200:])
    out = json.loads(r['stdout']); doc = out['receipt']
    rows = {r['arm']: r for r in doc['fitting_summary']}
    assert set(rows) == {'uncapped','risk_priority'}
    a, b = rows['uncapped'], rows['risk_priority']
    for key in ('sample_hash','query_draws','row_draws','first_monitor'):
        assert a[key] == b[key]
    assert a['step'] == b['step'] == 100 and a['query_draws'] == 3200
    assert a['unknown_rows_sampled'] == b['unknown_rows_sampled'] == 0
    assert b['min_risk_projection'] >= .5-1e-6
    projected = (doc['observed_fitting_seconds']*20+max(0, doc['seconds']-doc['observed_fitting_seconds']))*108
    out.update(result_source='fresh_run_real_training_verified', total_updates=200,
        projected_full_seconds_from_first_pair=projected,
        projection_is_not_measured_full_runtime=True, full_training_resource_check=projected < 6800 and doc['peak_RSS_KiB'] < 12*2**20,
        full_training='not_run_at_pilot_collection', held_readout='not_run', independent_confirmation='not_run',
        deployment_changed=False, stage5c_executed=False, smc_enabled=False,
        collector_sha256=digest(Path(__file__)))
    immutable(PUBLIC/'pilot_result.json', out)
    assert out['full_training_resource_check'], 'Explicit resource amendment needed; do not reduce science scope'
    print(json.dumps(dict(job=doc['job_id'], verified_files=2, total_updates=200, projected_full_seconds=projected, resource_check=True)))


if __name__ == '__main__':
    main()
