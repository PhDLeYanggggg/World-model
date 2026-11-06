"""Fit complete-risk donor heads without opening a new evaluation partition."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_complete_donor_policy as full
from src.world_model import m3w_complete_risk_donors as method
from src.world_model.m3w_floor_relative import causal_cost_features

np, torch = full.np, full.torch
read, stable, sha, once, checked, artifact = full.read, full.stable, full.sha, full.once, full.checked, full.artifact
donor, seeds, neural, source = full.donor, full.seeds, full.neural, full.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_complete_risk_donors_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_complete_risk_donors_v1'
CONFIG = ROOT/'configs/m3w_moving_complete_risk_donors_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_complete_risk_donors.py',
         ROOT/'tests/test_m3w_complete_risk_donors.py')


def register():
    cfg = read(CONFIG); _, parent = full.register()
    if (cfg['risk_head_seeds'] != [17, 29, 43] or cfg['fresh_models'] != 24 or cfg['steps_per_model'] != 4096
            or cfg['risk_budget'] != .02 or cfg['risk_threshold'] != 0 or cfg['augmentation_mass'] != .5
            or any(cfg[k] for k in ('threshold_search', 'new_outer_C_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed complete-risk treatment and unchanged original rules required')
    value = dict(config=cfg, jobs=parent['jobs'], files={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        settings=donor.method.SETTINGS, parent_registration=artifact(full.REG),
        full_inference=artifact(full.PUBLIC/'inference_freeze_v2.json'),
        prior_result=artifact(full.PUBLIC/'analysis.json'),
        A_donors=artifact(donor.PUBLIC/'donors.json'), A_forecasts=artifact(source.PUBLIC/'analysis.json'),
        parent_code={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(method.amplitude.__file__), Path(donor.method.__file__), Path(seeds.method.__file__),
             Path(neural.method.__file__), Path(full.method.original.__file__), Path(full.method.__file__))},
        role='already_exposed_source_excluded_complete_risk_training', independent_confirmation=False,
        original_rules_unchanged=True, new_outer_C_H_readout=False)
    once(REG, stable(value)); return cfg, value


def committed():
    for p in (*FILES, REG):
        body = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                              capture_output=True, check=True).stdout
        if hashlib.sha256(body).hexdigest() != sha(p):
            raise ValueError('Commit exact model treatment before real work')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume saved model')))
    started, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume same experiment after12h')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Keep5GiB reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('Private artifact cap exceeded')
    def beat(**values):
        guard()
        event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
        source.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', device='cpu', machine=platform.machine(), torch_threads=4, interop_threads=1, workers=0)
    return started, guard, beat


def load_cache(ref):
    with np.load(checked(ref), allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def prepare_own(reg, beat):
    data = full.load_data(); refs = []
    previous = read(neural.PUBLIC/'analysis.json')
    for node in reg['jobs']:
        job = node['job']; key = job['key']
        old = load_cache(read(full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
        ix, active = old['local_ids'], old['active']; paths = old['paths']
        scales = causal_cost_features(data['geometry'][ix], old['floor'], old['floor'])[1].astype(np.float32)
        packet = dict(paths=paths, floor=old['floor'], cv=old['cv'],
                      causal=dict(envelope=np.linalg.norm(paths-old['floor'][:, None], axis=-1).max(-1)))
        y, known = method.signed_targets(packet, data['target_eval'][ix]-data['origin'][ix, None],
                                        data['valid'][ix], job['easy_cut'], scales)
        choice = old['proposal_self'][active]
        own = dict(x=old['summary120'], target=y[active, choice], sites=old['sites'][active],
            ids=old['ids'][active], query=data['queries'][ix][active], observed_future=known[active])
        own['weights'] = donor.method.parent.base.balanced_weights(own['sites'], own['query'])
        c = next(c for c in previous['contexts'] if c['job']==key); frozen, _ = neural.load(c)
        mask = own['observed_future']
        for k in ('x', 'target', 'sites', 'ids'):
            np.testing.assert_array_equal(own[k][mask], frozen[k])
        path = PRIVATE/'own'/f'{key}.npz'; full.save_arrays(path, own)
        refs.append(dict(job=key, cache=artifact(path), rows=len(active), known=int(mask.sum()),
                         unknown=int((~mask).sum()), known_features_labels_exact=True))
        beat(state='complete_own_supervision_prepared', **{k:refs[-1][k] for k in ('job','rows','unknown')})
    once(PUBLIC/'own.json', stable(dict(banks=refs, new_future_input=False)))


def prepare_A(reg, cfg, guard, beat):
    data = source.load_data(); bank = read(source.PUBLIC/'analysis.json'); refs = []
    for fold in (0, 1):
        job = next(n['job'] for n in reg['jobs'] if n['job']['producer_fold']==fold)
        chunks = {k:[] for k in ('x','target','sites','query','ids')}
        for item in bank['forecasts']:
            if item['job']['producer_fold'] != fold: continue
            saved = load_cache(item['forecast']); ids = saved['local_ids']
            np.testing.assert_array_equal(data['global_ids'][ids], saved['global_ids'])
            if not set(data['sites'][ids]) <= set(job['producer_sites']):
                raise ValueError('Wrong parent-A donor scope')
            take = np.flatnonzero(~data['valid'][ids].any(1))
            for begin in range(0, len(take), 512):
                guard(); positions = take[begin:begin+512]; ix = ids[positions]
                forecast = saved['relative_prediction'][positions].astype(float)
                physical = np.stack([donor.baseline_numpy(data['history'][ix], k)-data['origin'][ix, None]
                                     for k in range(6)], 1)
                floor = physical[:, job['floor_index']]
                for action, alpha in enumerate(donor.FRACTIONS, 1):
                    proposal = floor+alpha*(forecast-floor) if alpha != 1 else forecast
                    x, scales, allowed, _ = donor.method.causal_features(data['geometry'][ix],
                        data['shared_context'][ix], physical, proposal, data['recordings'][ix], data['frames'][ix],
                        data['global_ids'][ix], data['sites'][ix], job['floor_index'])
                    y = method.unknown_targets(np.linalg.norm(proposal-floor, axis=-1).max(-1), scales)
                    for k, v in dict(x=x, target=y, sites=data['sites'][ix], query=data['queries'][ix],
                                     ids=data['global_ids'][ix]*7+action).items(): chunks[k].append(v[allowed])
            beat(state='unknown_A_forecast_prepared', forecast=item['job']['key'])
        unknown = {k:np.concatenate(v) for k,v in chunks.items()}
        complete = method.append_unknown(donor.load_added(fold), unknown)
        path = PRIVATE/'unknown'/f'A{fold}.npz'; full.save_arrays(path, unknown)
        refs.append(dict(producer_fold=fold, cache=artifact(path), unknown_rows=len(unknown['ids']),
            complete_rows=len(complete['ids']), known_rows=int(complete['observed_future'].sum()),
            complete_ids=neural.method.digest(complete['ids']), complete_targets=neural.method.digest(complete['target'])))
    if sum(r['unknown_rows'] for r in refs) != cfg['expected_A_unknown_candidates']:
        raise ValueError('All original omitted legal A candidates required')
    once(PUBLIC/'unknown_A.json', stable(dict(banks=refs, bounds_not_observed_labels=True)))


def entries(reg):
    for node in reg['jobs']:
        job = node['job']; key = job['key']
        own_ref = next(r for r in read(PUBLIC/'own.json')['banks'] if r['job']==key)
        own = load_cache(own_ref['cache'])
        ref = next(r for r in read(PUBLIC/'unknown_A.json')['banks'] if r['producer_fold']==job['producer_fold'])
        added = method.append_unknown(donor.load_added(job['producer_fold']), load_cache(ref['cache']))
        for field, digest in (('ids','complete_ids'), ('target','complete_targets')):
            if neural.method.digest(added[field]) != ref[digest]: raise ValueError('Changed complete A donors')
        for site in node['part']['names']:
            d = donor.method.combined(own, added, site, 'augmented')
            for seed in (17,29,43):
                identity = dict(registration_sha256=sha(REG), job=key, source=site, risk_head_seed=seed,
                    input_hash=neural.method.digest(d['x']), target_hash=neural.method.digest(d['target']),
                    weights_hash=neural.method.digest(d['weights']), complete_risk_supervision=True)
                yield node, site, seed, d, identity


def fit_path(job, site, seed, kind='fits'):
    return PRIVATE/kind/f'{job}_{site}_seed{seed}.pt.gz'


def train(reg, guard, beat, resume=False, verify=False):
    refs = []
    for node, site, seed, d, identity in entries(reg):
        job = node['job']['key']; path = fit_path(job, site, seed, 'verify' if verify else 'fits')
        state = seeds.method.fit(d, site, identity, path, seed=seed, resume=resume,
            guard=guard, heartbeat=lambda **kw:beat(job=job, source=site, seed=seed, **kw))
        if state['step']!=4096 or not state['complete'] or site in state['preprocess']['train_sources']:
            raise ValueError('Complete source-excluded fixed budget required')
        if verify:
            donor.method.exact(neural.rel.base.read_checkpoint(fit_path(job,site,seed)), state)
        else:
            out = PUBLIC/'fits'/f'{job}_{site}_seed{seed}.json'
            once(out, stable(dict(job=job, source=site, seed=seed, checkpoint=artifact(path),
                steps=state['step'], fit_rows=state['fit_rows'], fitting_sources=state['preprocess']['train_sources'],
                seconds=state['seconds'], trace=state['trace'], result_source='fresh_run')))
            refs.append(artifact(out))
        beat(state='full_refit_exact' if verify else 'fit_complete', job=job, source=site, seed=seed)
    if verify: once(PUBLIC/'verification.json', dict(all24_full_refits_exact=True, repeat_updates=98304))
    else: once(PUBLIC/'training_freeze.json', dict(fits=refs, new_models=len(refs), primary_updates=98304,
                                                all_before_readout=True, final_checkpoint_only=True))


def infer(reg, beat, replay=False):
    if (read(PUBLIC/'training_freeze.json')['new_models']!=24
            or not read(PUBLIC/'verification.json')['all24_full_refits_exact']):
        raise ValueError('All fits/refits before outcomes')
    refs = []
    for node in reg['jobs']:
        key=node['job']['key']; old=load_cache(read(full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
        active=old['active']; prediction=np.zeros((3,len(old['ids']),2))
        for site in node['part']['names']:
            at=np.flatnonzero(old['sites'][active]==site)
            for si,seed in enumerate((17,29,43)):
                record=read(PUBLIC/'fits'/f'{key}_{site}_seed{seed}.json')
                state=neural.rel.base.read_checkpoint(checked(record['checkpoint']))
                if site in state['preprocess']['train_sources']:raise ValueError('Source leakage')
                prediction[si,active[at]]=neural.method.predict(state,old['summary120'][at])
        proposal=old['proposal_self']; aggregated=full.scores(prediction)
        choices={f'seed{seed}':full.method.veto(proposal,prediction[si]) for si,seed in enumerate((17,29,43))}
        choices.update({name:full.method.veto(proposal,aggregated[name]) for name in ('mean','unanimity')})
        payload=dict(risk=prediction,**choices); path=PRIVATE/'inference'/f'{key}.npz'
        if replay:
            cached=load_cache(read(PUBLIC/'inference'/f'{key}.json')['cache'])
            for k,v in payload.items():np.testing.assert_array_equal(cached[k],v)
        else:
            full.save_arrays(path,payload)
            once(PUBLIC/'inference'/f'{key}.json',dict(cache=artifact(path),choices_before_labels=True))
        refs.append(artifact(PUBLIC/'inference'/f'{key}.json'))
        beat(state='complete_causal_choices_frozen',job=key)
    if not replay:once(PUBLIC/'inference_freeze.json',dict(contexts=refs,all_before_readout=True))


def evaluate(reg,cfg,beat,replay=False):
    if not read(PUBLIC/'inference_freeze.json')['all_before_readout']:raise ValueError('Freeze all choices first')
    data=full.load_data(); rows=[]
    for node in reg['jobs']:
        job=node['job']; key=job['key']
        old=load_cache(read(full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
        new=load_cache(read(PUBLIC/'inference'/f'{key}.json')['cache'])
        ix=old['local_ids']; target=data['target_eval'][ix]-data['origin'][ix,None]; valid=data['valid'][ix]
        paths={name:full.method.paths_for(old['paths'],old['floor'],new[name])
               for name in ('seed17','seed29','seed43','mean','unanimity')}
        for site in node['part']['names']:
            at=old['sites']==site
            policies={name:full.method.original.policy(p[at],old['floor'][at],old['cv'][at],target[at],valid[at],
                old['recordings'][at],easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in paths.items()}
            contrasts={name:full.method.original.paired(full.method.paths_for(old['paths'],old['floor'],old[name])[at],
                paths[name][at],old['floor'][at],target[at],valid[at]) for name in ('mean','unanimity')}
            rows.append(dict(job=key,source=site,policies=policies,paired_vs_known_only=contrasts))
        beat(state='original_complete_policy_evaluated',job=key)
    policies={name:dict(safety=full.method.safety([r['policies'][name] for r in rows]),
        source_macro_known_ADE_gain=float(np.mean([r['policies'][name]['known_ADE_gain_floor_percent'] for r in rows])),
        source_macro_completion_lower_gain=float(np.mean([r['policies'][name]['completion_lower_gain_floor_percent'] for r in rows])),
        selected=sum(round(r['policies'][name]['switch_rate']*r['policies'][name]['rows']) for r in rows),
        unknown_selected=sum(v['unknown_selected'] for r in rows for v in r['policies'][name]['recordings'].values()))
        for name in paths}
    pick=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(rows),(cfg['bootstrap_resamples'],len(rows)))
    contrasts={}
    for name in ('mean','unanimity'):
        contrasts[name]={}
        for key in ('known_pp','completion_lower_pp','completion_upper_pp'):
            values=[r['paired_vs_known_only'][name][key] for r in rows]
            contrasts[name][key]=dict(point=float(np.mean(values)),interval95=np.quantile(np.asarray(values)[pick].mean(1),[.025,.975]).tolist())
    result=stable(dict(rows=rows,policies=policies,paired_vs_known_only=contrasts,
        result_source='fresh_run',fresh_models=24,risk_seeds=[17,29,43],forecaster_seeds=[17],
        original_rules_unchanged=True,original24group_primary='not_estimable_not_passed',
        new_outer_C_H_readout=False,identity_certification=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay:neural.rel.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','prepare','train','verify','infer','evaluate','replay'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Complete-risk model treatment registered.');return
        committed();start,guard,beat=runtime(cfg,args.phase)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare_own(reg,beat);prepare_A(reg,cfg,guard,beat)
        elif args.phase in ('train','verify'):train(reg,guard,beat,resume=args.resume,verify=args.phase=='verify')
        elif args.phase=='infer':infer(reg,beat)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat)
        else:
            infer(reg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            neural.rel.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        once((PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json'),
            dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        beat(state='phase_complete')


if __name__=='__main__':main()
