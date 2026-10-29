"""Registered TRAIN-only local quadratic-risk fit with immutable queries."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch, not x86/Rosetta')

from scripts import run_m3w_selection_conditioned_analogue as parent
from src.evaluation import m3w_local_cost_analogue as api

ROOT, sha, once = parent.ROOT, parent.sha, parent.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_local_cost_analogue_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_local_cost_analogue_v1'
REG = PUBLIC/'registration.json'
pooled = parent.parent


def register():
    reg, cfg, ancestor = parent.admission()
    files = [Path(__file__), ROOT/'src/evaluation/m3w_local_cost_analogue.py',
        ROOT/'tests/test_m3w_local_cost_analogue.py', PUBLIC/'protocol.md',
        PUBLIC/'prior_controls.md', parent.REG, parent.PUBLIC/'complete.json',
        parent.PUBLIC/'summary.json', parent.PUBLIC/'conclusions.md']
    once(REG, dict(bindings={str(p.resolve().relative_to(ROOT)):sha(p) for p in files},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], views=72, k=32,
        primary='local_signed32', changed_factor='local_cost_objective_with_equal_source_weight_and_frozen_reference_offset',
        normalization_changed=False, query_policy_changed=False, shuffle_seed=20261029,
        held_inference=False, independent_roles_read=False, neural_updates=0,
        whole_pipeline_OOF=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    result = register()
    for rel in json.loads(REG.read_text())['bindings']:
        pooled.previous.previous.committed(ROOT/rel)
    pooled.previous.previous.committed(REG)
    return result


def runtime():
    pooled.previous.previous.train.torch.set_num_threads(4)
    pooled.previous.previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    def beat(**kw):
        value = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        p = PRIVATE/'heartbeat.tmp'; p.write_text(json.dumps(value)+'\n'); os.replace(p, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        if kw.get('state') != 'analogue_queries': print(json.dumps(value), flush=True)
        if time.monotonic()-start > 43200: raise TimeoutError('12h boundary; resume unchanged protocol')
        if shutil.disk_usage(ROOT).free < 10*2**30: raise OSError('Preserve10GiB reserve')
    beat(state='admitted')
    return start, beat


def diagnostic(out, take, y, p, env, scale):
    support = out['supported'][take]; known = np.isfinite(y).all(1) & support
    result = dict(supported_queries=int(support.sum()), known_queries=int(known.sum()),
        effective_donor_quantiles=np_quantiles(out['diagnostics']['effective_donors'][take][support]),
        max_weight_quantiles=np_quantiles(out['diagnostics']['max_weight'][take][support]),
        donor_loss={key:out['donor_loss'][key][take][support].sum(0).tolist() for key in ('uniform', 'optimum')},
        donor_loss_is_training_objective_not_generalization=True, arms={})
    for arm, pred in p.items():
        result['arms'][arm] = dict(easy_exceeds_all=int((pred[support,4] > pred[support,1]+1e-7).sum()))
        for name, col in (('all',1), ('easy',4)):
            error = ((pred[known,col]-y[known,col])/scale)**2
            event = y[known,col] > 0
            result['arms'][arm][name] = dict(events=int(event.sum()),
                event_harm_SSE=float(error[event].sum()), non_event_harm_SSE=float(error[~event].sum()),
                predicted_harm_on_events=float(pred[known,col][event].sum()/scale),
                predicted_harm_on_non_events=float(pred[known,col][~event].sum()/scale))
    return result


def np_quantiles(values):
    return api.np.quantile(values,[0,.25,.5,.75,1]).tolist() if len(values) else None


def execute(phase, resume=False):
    reg, _, ancestor = admission()
    states = pooled.previous.previous.models(reg)
    _, origins = pooled.previous.previous.parent.models(reg)
    previous = json.loads((parent.PUBLIC/'complete.json').read_text())
    refs_old = {Path(r['path']).name:r for r in previous['groups']}
    pilot, check = phase == 'pilot', phase == 'check'
    finish = PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not check: raise ValueError('Completed phase immutable')
    if phase == 'run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot must pass first')
    start, beat = runtime(); rows = []; receipts = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        contexts = sorted(reg['contexts'].items())
        for context, refs in contexts[:1] if pilot else contexts:
            packets, held = pooled.LOADER.packets(refs, ancestor, beat); del held
            bank, spans, _ = pooled.api.pool(packets)
            bank['scale'] = api.np.concatenate([api.np.full(len(packets[s][0]), packets[s][-1]['scale']) for s in sorted(packets)])
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                paths = {s:PUBLIC/'groups'/f'{context}_{s}_seed{seed}.json' for s in sorted(packets)}
                if resume and not check and not pilot and all(p.exists() for p in paths.values()):
                    fresh = [json.loads(p.read_text()) for p in paths.values()]
                    if any(r['registration_sha256'] != sha(REG) for r in fresh): raise ValueError('Resume binding changed')
                else:
                    values, queries, _ = pooled.predictions(context, seed, packets, origins, states)
                    q = api.np.concatenate([queries[s] for s in sorted(packets)])
                    frozen = api.np.concatenate([values[s]['origin'] for s in sorted(packets)])
                    out = api.retrieve(bank, q, frozen, heartbeat=beat); fresh = []
                    for source in sorted(packets):
                        oldpath = parent.PUBLIC/'groups'/paths[source].name
                        if sha(oldpath) != refs_old[oldpath.name]['sha256']: raise ValueError('Parent result changed')
                        old = json.loads(oldpath.read_text())
                        lo,hi = spans[source]; take = (out['indices']>=lo)&(out['indices']<hi)
                        ix = out['indices'][take]-lo; _,env,y,_,_,_,_,pr = packets[source]
                        p = {'origin':values[source]['origin'][ix]}
                        for arm in api.MODES:
                            p[arm] = p['origin'].copy(); p[arm][:,[1,4]] = out['predictions'][arm][take]
                        references = [r for r in out['references'] if r['query_source']==source]
                        if len(references) != len(old['references']): raise ValueError('Reference population changed')
                        for a,b in zip(references,old['references']):
                            fields = ('query_sha256','donor_sha256','known_donors','causal_donors','supported',
                                      'known_donors_by_source') + (('mean_sha256','std_sha256','neighbor_sha256') if a['supported'] else ())
                            if any(a[k]!=b[k] for k in fields): raise ValueError('Frozen reference or neighbor changed')
                        for arm in ('origin','selected_nearest32'):
                            if api.base.fingerprint(p[arm]) != old['prediction_sha256'][arm]: raise ValueError('Inherited prediction changed')
                        if pilot: continue
                        support = out['supported'][take]
                        scores = {a:api.base.score(y[ix],v,env[ix],pr['scale'],support if a!='origin' else api.np.ones(len(ix),bool)) for a,v in p.items()}
                        for arm in ('origin','selected_nearest32'):
                            if scores[arm] != old['scores'][arm]: raise ValueError('Inherited score changed')
                        row = dict(context=context,source=source,seed=seed,scope='fit',
                            registration_sha256=sha(REG),parent_group_sha256=sha(oldpath),
                            references=references, scores=scores,
                            prediction_sha256={a:api.base.fingerprint(v) for a,v in p.items()},
                            supported_sha256=api.base.fingerprint(support),
                            diagnostics=diagnostic(out,take,y[ix],p,env[ix],pr['scale']),
                            result_source='fresh_run_closed_form_local_cost_fit',neural_updates=0,
                            whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,deployment_changed=False)
                        if check:
                            if row != json.loads(paths[source].read_text()): raise ValueError('Full replay differs')
                        else: once(paths[source],row)
                        fresh.append(row)
                    if pilot:
                        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            views=4,queries=len(out['indices']),supported=int(out['supported'].sum()),
                            frozen_predictions_exact=True,neighbors_and_normalization_exact=True,
                            engineering_pass=bool(out['supported'].all()),scientific_screen_evaluated=False,
                            registration_sha256=sha(REG)))
                        beat(state='pilot_complete'); return
                rows.extend(fresh)
                receipts.extend(dict(path=str(paths[r['source']].relative_to(ROOT)),sha256=sha(paths[r['source']])) for r in fresh)
                beat(state='verified_views' if check else 'fit_views',views=len(rows),total=72)
        summary = api.summarize(rows,pooled.previous.previous.parent.expected(reg))
        if check:
            done = json.loads(finish.read_text())
            if done['groups']!=receipts or done['summary_sha256']!=sha(PUBLIC/'summary.json') or summary!=json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(finish,dict(groups=receipts,registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json'),
                pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0,held_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if check else 'run_complete',magnitude_screen=summary['magnitude_screen_pass'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('register','pilot','run','check'))
    parser.add_argument('--resume',action='store_true'); args = parser.parse_args()
    if args.action == 'register': register()
    else: execute(args.action,args.resume)
