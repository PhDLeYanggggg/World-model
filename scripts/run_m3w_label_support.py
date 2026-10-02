"""Frozen source-action and raw-label diagnostic; no new model or policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import zipfile

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_leaf_geometry as leaf
from scripts import audit_m3w_european_observation_quality as observation
from src.world_model import m3w_label_support_diagnostic as api

parent, sha, once = leaf.parent, leaf.sha, leaf.once
NAME = 'european_label_support_v1'
PUBLIC = leaf.PUBLIC.parent/NAME
PRIVATE = leaf.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')


def registration():
    assert leaf.registration() == json.loads((leaf.PUBLIC/'registration.json').read_text())
    obs = json.loads((observation.PUBLIC/'verification.json').read_text())
    for p,h in obs['source_bindings'].items(): assert sha(ROOT/p) == h
    for p,h in obs['artifacts'].items(): assert sha(observation.PUBLIC/p) == h
    observation.registration()
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_label_support'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_label_support_diagnostic.py']
    return dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        leaf_verification_sha256=sha(leaf.PUBLIC/'verification.json'),
        observation_verification_sha256=sha(observation.PUBLIC/'verification.json'),
        training=False, independent_roles_read=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def quality(data, cfg, pilot):
    manifest, roles, packed, _ = observation.registration()
    agents_path = observation.PACKED/'agents.npy'
    assert observation.digest(agents_path) == packed['arrays']['agents']
    agents = np.load(agents_path, mmap_mode='r', allow_pickle=False)
    n = len(agents); out = {k:np.full(n, np.nan) for k in cfg['past_diagnostics']+cfg['future_diagnostics_offline_only']}
    seen = np.zeros(n, bool); receipts = []
    prior = json.loads((observation.PUBLIC/'audit.json').read_text())
    src = json.loads((observation.BASE/'european_squares_intake_v1/trajectory_manifest.json').read_text())['private_file']
    archive = ROOT/src['path']; beat(state='verify_raw_archive', bytes=src['bytes'])
    assert archive.stat().st_size == src['bytes'] and observation.digest(archive) == src['sha256']
    with zipfile.ZipFile(archive) as z:
        for ri,ref in enumerate(manifest['record_receipts']):
            access = observation.require_source_training(roles, ref['source_member'])
            home = ROOT/ref['directory']; assert sha(home/'receipt.json') == ref['receipt_sha256']
            receipt = json.loads((home/'receipt.json').read_text())
            assert receipt['rows_sha256'] == access['rows_sha256']
            old_ref = prior['receipts'][ri]; assert sha(ROOT/old_ref['path']) == old_ref['sha256']
            old = json.loads((ROOT/old_ref['path']).read_text())
            assert old['source'] == ref and observation.artifact(ROOT/old['diagnostics']['path']) == old['diagnostics']
            ids = np.flatnonzero(data['recordings'] == ri)
            assert parent.base.inter.array_hash(ids) == old['packed_ids_sha256'] and not seen[ids].any()
            with np.load(ROOT/old['diagnostics']['path'], allow_pickle=False) as saved:
                for key in cfg['past_diagnostics']:
                    assert saved[key].shape == (len(ids),)
                    out[key][ids] = saved[key]
            with z.open(ref['source_member']) as f: rows,_ = observation.read_raw_csv(f)
            assert hashlib.sha256(rows.tobytes()).hexdigest() == receipt['rows_sha256']
            raw_agents, starts, sizes = np.unique(rows['agent'], return_index=True, return_counts=True)
            covered = 0
            for agent,start,size in zip(raw_agents,starts,sizes):
                at = ids[agents[ids] == agent]
                if not len(at): continue
                d = api.raw_future_quality(rows[start:start+size], data['frames'][at],data['target_eval'][at],data['valid'][at])
                for key in cfg['future_diagnostics_offline_only']: out[key][at] = d[key]
                covered += len(at)
            assert covered == len(ids); seen[ids] = True
            receipts.append(dict(index=ri, source_receipt_sha256=ref['receipt_sha256'],
                past_diagnostic_sha256=old['diagnostics']['sha256'], rows=len(ids),
                valid_label_boxes=int(data['valid'][ids].sum()), raw_rows_sha256=receipt['rows_sha256'],
                diagnostics_hash={k:parent.base.inter.array_hash(v[ids]) for k,v in out.items()}))
            if ri % 10 == 0: beat(state='raw_labels_matched', recordings=ri+1, rows=int(seen.sum()))
    assert seen.all() and len(receipts) == 163 and n == 318969
    return out, agents, dict(records=receipts, raw_archive_sha256=src['sha256'], rows=n,
        valid_label_boxes=int(data['valid'].sum()), future_masks_and_coordinates_exact=True,
        identity_is_tracker_key_not_verified_physical_identity=True)


def summarize(groups, unique, cfg):
    strata = {}
    for name in groups[0]['cohort']['strata']:
        values = [g['cohort']['strata'][name] for g in groups]
        total = {k:sum(v[k] for v in values) for k in values[0] if k != 'easy_harm_ratio'}
        total['easy_harm_ratio'] = total['easy_harm']/total['easy_reference'] if total['easy_reference'] > 0 else None
        strata[name] = total
    def interval(f): return api.locality_interval(groups,f,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    def share(g):
        s = g['cohort']['strata']; h = sum(v['harm'] for v in s.values())
        return s['all_twelve']['harm']/h if h > 0 else None
    contrasts = {k:{m:interval(lambda g,k=k,m=m:g['contrasts'][k][m])
                     for m in ('raw_mean_difference','query_matched_mean_difference')}
                 for k in cfg['past_diagnostics']+cfg['future_diagnostics_offline_only']}
    return dict(groups=len(groups), source='fresh_run_offline_association_cached_verified_models',
        repeated_occurrence_strata=strata, unique=unique,
        full_label_share_of_selected_harm=interval(share), contrasts=contrasts,
        annotations='detector_silver_not_human_gold', causal_noise_attribution=False,
        new_training=False, deployment_changed=False, independent_roles_read=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register','pilot','run','verify']); args = p.parse_args()
    reg = registration(); cfg = json.loads(CONFIG.read_text())
    if args.phase == 'register':
        once(PUBLIC/'registration.json',reg); print(json.dumps(dict(registered=True))); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify': assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    began = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _,_,data,jobs,oid,_,_,_ = parent.inner.old.load()
        qa,agents,raw_receipt = quality(data,cfg,args.phase == 'pilot')
        docs,cal = parent.parent.docs(),parent.docs()
        groups,refs = [],[]; selected_ids=set(); harm_ids=set(); all_ids=set(); checks=0; bytes_=0
        pilot = args.phase == 'pilot'
        for c in parent.parent.contexts(data,jobs,oid):
            for site in parent.inner.sources(c):
                group = c['name']+'_fit_'+site
                at,ids,x,env,y,_,upstream = parent.inner.training_arrays(c,data,site)
                _,val,partition = parent.forest.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
                vid, yv, ev = ids[val],y[val],env[val]
                temporal = api.temporal_errors(c['floor'][at][val].astype(float)+data['origin'][vid,None],
                    c['prediction'][at][val].astype(float)+data['origin'][vid,None],data['target_eval'][vid],data['valid'][vid])
                known = np.isfinite(yv).all(1)
                np.testing.assert_allclose(temporal['signed_error'][known],yv[known,1]-yv[known,0],rtol=1e-9,atol=1e-8)
                for seed in cfg['head_seeds']:
                    assert time.monotonic()-began < cfg['hard_runtime_limit_seconds']
                    old,ca = docs[group,seed],cal[group,seed]
                    state = joblib.load(ROOT/old['checkpoint']['path'])
                    assert state['identity']['upstream'] == upstream and old['partition'] == partition
                    pred,support = parent.forest.api.predict(state,x[val],ev)
                    again,sup2 = parent.forest.api.predict(state,x[val],ev)
                    np.testing.assert_array_equal(pred,again); np.testing.assert_array_equal(support,sup2)
                    h = parent.base.inter.array_hash
                    assert h(vid) == old['validation']['ids_hash'] == ca['identity']['validation_ids_hash']
                    assert h(pred) == old['validation']['prediction_hashes']['forest'] == ca['identity']['prediction_hash']
                    assert h(yv) == ca['identity']['target_hash'] and h(ev) == ca['identity']['envelope_hash']
                    take = parent.api.eligible(pred,c['moving'][at][val],support)
                    assert h(take) == old['validation']['action_hashes']['forest']
                    checks += 7
                    harm = take & known & (yv[:,1] > 0); safe = take & known & ~harm
                    co = api.cohort(yv,take,ev,vid,data['recordings'][vid],agents[vid],temporal['valid_steps'],temporal)
                    con = {k:api.contrasts(v[vid],harm,safe,data['recordings'][vid],data['frames'][vid]) for k,v in qa.items()}
                    item = dict(source=site,group=group,head_seed=seed,partition=partition,
                        checkpoint=old['checkpoint'],cohort=co,contrasts=con,
                        hashes=dict(ids=h(vid),prediction=h(pred),action=h(take),target=h(yv),temporal={k:h(v) for k,v in temporal.items()}))
                    groups.append(item); selected_ids.update(vid[take].tolist()); harm_ids.update(vid[harm].tolist()); all_ids.update(vid.tolist())
                    dest = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                    if not pilot:
                        once(dest,item); bytes_ += dest.stat().st_size
                        assert bytes_ < cfg['aggregate_output_cap_bytes']
                        refs.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
                    beat(state='group_complete',heads=len(groups),group=group,head_seed=seed,seconds=time.monotonic()-began)
                    if pilot: break
                if pilot: break
            if pilot: break
        def counts(ids):
            at = np.array(sorted(ids),int)
            return dict(rows=len(at),recording_queries=len(set(zip(data['recordings'][at].tolist(),data['frames'][at].tolist()))),
                        tracks=len(set(zip(data['recordings'][at].tolist(),agents[at].tolist()))),
                        recordings=len(np.unique(data['recordings'][at])),ids_hash=parent.base.inter.array_hash(at))
        unique = {k:counts(v) for k,v in dict(validation=all_ids,selected=selected_ids,harmful_at_least_one_head=harm_ids).items()}
        result = summarize(groups,unique,cfg)
        runtime = dict(pid=os.getpid(),seconds=time.monotonic()-began,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            native_architecture=platform.machine(),cpu_threads=4,num_workers=0,
            available_disk_bytes=shutil.disk_usage(ROOT).free,cache_reserve_bytes=cfg['cache_reserve_bytes'],
            numerical_cache_written=False,checkpoint_written=False,new_HPC_jobs=0,head_hash_checks=checks,
            exact_inference_replay=True,groups=refs)
        if pilot: once(PUBLIC/'pilot.json',dict(runtime=runtime,summary=result,raw_receipt=raw_receipt))
        else:
            assert len(groups) == cfg['source_heads']
            once(PUBLIC/'raw_label_receipt.json',raw_receipt)
            once(PUBLIC/'summary.json',result)
            if args.phase == 'verify':
                once(PUBLIC/'replay.json',dict(runtime,raw_and_readout_exact=True,summary_sha256=sha(PUBLIC/'summary.json')))
            else: once(PUBLIC/'complete.json',dict(runtime,summary_sha256=sha(PUBLIC/'summary.json'),raw_receipt_sha256=sha(PUBLIC/'raw_label_receipt.json')))
        beat(state='complete',phase=args.phase,seconds=time.monotonic()-began)


if __name__ == '__main__': main()
