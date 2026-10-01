"""Storage-only amendment: resume frozen packets with a 2 GiB remote cap."""
import argparse
import io
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import manage_m3w_selected_pool_create as manager
from scripts.manage_m3w_boundary_diagnostic import RECEIVER
from scripts.m3w_packet_stream import PacketStream

run = manager.run
PUBLIC, PRIVATE, REMOTE = manager.PUBLIC, manager.PRIVATE, manager.REMOTE


def registration():
    assert manager.registration() == json.loads((PUBLIC/'recovery_registration.json').read_text())
    return dict(original_recovery_sha256=run.digest(PUBLIC/'recovery_registration.json'),
        code_sha256=run.digest(__file__), note_sha256=run.digest(PUBLIC/'storage_amendment.md'),
        prior_cap_bytes=512*2**20, input_byte_cap=2*2**30, groups=216,
        reason='prior_remote_input_cap_exceeded_before_submission',
        local_array_cache=False, personal_quota_verified=False,
        method_changed=False, actions_changed=False, risk_budget_changed=False)


def export(amendment):
    reg = manager.registration()
    setup = r'''
import json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root==pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
assert json.loads((root/'recovery_registration.json').read_text())==p['original']
assert not (root/'submission_intent.json').exists()
path=root/'storage_amendment.json';text=json.dumps(p['amendment'],indent=2)+'\n'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
print(json.dumps({'existing_packets':len(list((root/'inputs').glob('*.npz')))}))
'''
    print(json.dumps(manager.remote(setup, dict(root=REMOTE, original=reg, amendment=amendment))), flush=True)
    ssh = json.loads((manager.HANDOFF/'observations.json').read_text())['ssh_arguments']
    stream = PacketStream.__new__(PacketStream)
    stream.command = ssh+[shlex.join(['/usr/bin/python3', '-c', RECEIVER, REMOTE, run.NAME])]
    stream.process = None
    run.core.torch.set_num_threads(4); run.core.torch.set_num_interop_threads(1)
    _, _, data, jobs, oid, _, _, _ = run.inner.old.load(); fitted = run.parent.docs()
    frozen = {r['view']:r for r in json.loads((run.parent.PUBLIC/'decision_freeze.json').read_text())['rows']}
    old = {(r['view'],r['policy']):r['metric'] for r in json.loads((run.parent.PUBLIC/'readout.json').read_text())['rows']}
    refs = []; total = 0; parity = 0; started = time.monotonic()
    with stream:
        for c, at, ids, predictions, _, pr, oldmeta in run.parent.parent.views(data, jobs, oid):
            support = run.forest.api.causal_inputs(c['x'][at], c['env'][at], pr)[1]
            env, moving, rec = c['env'][at], c['moving'][at], data['recordings'][ids].astype(str)
            for seed in run.parent.parent.api.SEEDS:
                group = c['name']+'_fit_'+oldmeta['source']; view = oldmeta['view']+'_head'+str(seed)
                doc, fr = fitted[group,seed], frozen[view]; p = predictions[seed]
                raw = run.parent.api.eligible(p,moving,support)
                assert run.base.inter.array_hash(p) == fr['raw_prediction_hash']
                assert run.base.inter.array_hash(raw) == fr['action_hashes']['raw']
                arrays = dict(p=p,env=env,raw=raw,recordings=rec,
                    calibration_support=np.full(len(ids),doc['final']['supported'],bool))
                metas = {}
                for mode in run.parent.api.MODES:
                    q = run.parent.api.adjust(p,env,doc['final'],mode)
                    action = run.parent.api.eligible(q,moving,support)
                    assert run.base.inter.array_hash(q) == fr['adjusted_hashes'][mode]
                    assert run.base.inter.array_hash(action) == fr['action_hashes'][mode]
                    arrays[mode+'_q'] = q; arrays[mode+'_action'] = action
                    metas[mode] = dict(view=view,group=group,source=oldmeta['source'],site=oldmeta['site'],
                        head_seed=seed,mode=mode,role='transfer',
                        source_screen=doc['source'][mode]['oof']['finite_completion_supported'])
                cv, _, (floor,_), (neural,_) = run.base.floor_api.costs(c,data,at)
                arrays['y'] = run.core.targets(cv,floor,neural,c['job']['design']['easy_cut'])
                identity = dict(registration_sha256=run.digest(PUBLIC/'registration.json'),
                    parent_frozen_action=fr,ids_hash=run.base.inter.array_hash(ids))
                meta = dict(identity=identity, rows=metas,
                    parent_metrics={k:{f:old[view,k][f] for f in ('selected_positive_harm_ratio','selected_easy_positive_harm_ratio')}
                        for k in ('raw','harm','reference','joint')})
                local = PRIVATE/'transfer'/(view+'.json')
                if local.exists():
                    meta['expected_local'] = json.loads(local.read_text())
                    assert meta['expected_local']['identity'] == identity; parity += 1
                arrays['meta_json'] = np.array(json.dumps(meta,sort_keys=True))
                buf = io.BytesIO(); np.savez_compressed(buf,**arrays); packet = buf.getvalue()
                total += len(packet); assert total <= amendment['input_byte_cap']
                refs.append(stream.send(view,packet))
                if len(refs)%18 == 0: run.beat(state='CREATE_input_stream_v2',groups=len(refs),bytes=total)
    assert len(refs) == 216
    manifest = dict(packets=refs,bytes=total,local_parity_groups=parity,
        recovery_registration_sha256=run.digest(PUBLIC/'recovery_registration.json'),
        storage_amendment_sha256=run.digest(PUBLIC/'storage_amendment.json'))
    seal = r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
assert hashlib.sha256((root/'storage_amendment.json').read_bytes()).hexdigest()==p['manifest']['storage_amendment_sha256']
for r in p['manifest']['packets']:
    f=root/'inputs'/(r['group']+'.npz');assert f.stat().st_size==r['bytes']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
f=root/'input_manifest.json';text=json.dumps(p['manifest'],indent=2)+'\n'
if f.exists():assert f.read_text()==text
else:f.write_text(text)
print(json.dumps({'verified_packets':len(p['manifest']['packets'])}))
'''
    manager.remote(seal,dict(root=REMOTE,manifest=manifest))
    run.immutable(PUBLIC/'create_input_manifest.json',manifest)
    run.immutable(PUBLIC/'create_export_receipt.json',dict(groups=216,bytes=total,seconds=time.monotonic()-started,
        no_local_array_cache=True,remote_compute='not_run',local_parity_groups=parity,
        storage_amendment_sha256=manifest['storage_amendment_sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=['register','export']); a = p.parse_args()
    amendment = registration()
    if a.phase == 'register':
        run.immutable(PUBLIC/'storage_amendment.json',amendment); print('Storage amendment registered'); return
    assert json.loads((PUBLIC/'storage_amendment.json').read_text()) == amendment
    run.base.inter.committed(PUBLIC/'storage_amendment.json')
    export(amendment)


if __name__ == '__main__': main()
