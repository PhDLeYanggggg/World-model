"""Run a registered donor-selection repair without changing the query policy."""
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

from scripts import run_m3w_pooled_analogue as parent
from src.evaluation import m3w_selection_conditioned_analogue as api

ROOT, sha, once = parent.ROOT, parent.sha, parent.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selection_conditioned_analogue_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selection_conditioned_analogue_v1'
REG = PUBLIC/'registration.json'


def register():
    reg, cfg, ancestor = parent.admission()
    files = [Path(__file__), ROOT/'src/evaluation/m3w_selection_conditioned_analogue.py',
        ROOT/'tests/test_m3w_selection_conditioned_analogue.py', PUBLIC/'protocol.md',
        PUBLIC/'prior_controls.md', parent.REG, parent.PUBLIC/'complete.json',
        parent.PUBLIC/'summary.json', ROOT/'src/evaluation/m3w_selected_analogue.py']
    once(REG, dict(bindings={str(p.resolve().relative_to(ROOT)): sha(p) for p in files},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], views=72, k=32,
        changed_factor='label_donors_conditioned_on_same_frozen_policy',
        normalization_changed=False, query_policy_changed=False, shuffle_seed=20261029,
        held_inference=False, independent_roles_read=False, new_neural_training=False,
        whole_pipeline_OOF=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    result = register()
    for rel in json.loads(REG.read_text())['bindings']:
        parent.previous.previous.committed(ROOT/rel)
    parent.previous.previous.committed(REG)
    return result


def runtime():
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Native arm64 required')
    parent.previous.previous.train.torch.set_num_threads(4)
    parent.previous.previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    def beat(**kw):
        d = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(d)+'\n'); os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(d)+'\n')
        if kw.get('state') != 'analogue_queries': print(json.dumps(d), flush=True)
        if time.monotonic()-start > 43200: raise TimeoutError('12h limit; resume unchanged protocol')
        if shutil.disk_usage(ROOT).free < 10*2**30: raise OSError('Preserve10GiB reserve')
    return start, beat


def verify_full(bank, queries, out, oldrows):
    for source, old in oldrows.items():
        take = bank['sources'][out['indices']] == source
        for mode, arm in zip(api.MODES, api.ARMS[2:5]):
            r = old['prediction_sha256'][arm]
            # The old hash is over all five moments; the runner checks it below.
            if len(r) != 64: raise ValueError('Missing frozen prediction hash')
        refs = [r for r in out['references'] if r['query_source'] == source]
        if len(refs) != len(old['references']) or int(take.sum()) != old['scores']['origin']['rows']:
            raise ValueError('Frozen reference/query count changed')
        for a, b in zip(refs, old['references']):
            if any(a[k] != v for k, v in b.items()): raise ValueError('Full reference replay differs')
    if not api.np.array_equal(out['indices'], api.np.flatnonzero(queries)):
        raise ValueError('Query order changed')


def execute(phase, resume=False):
    reg, _, ancestor = admission()
    states = parent.previous.previous.models(reg)
    _, origins = parent.previous.previous.parent.models(reg)
    olddone = json.loads((parent.PUBLIC/'complete.json').read_text())
    oldrefs = {Path(r['path']).name: r for r in olddone['groups']}
    pilot, checking = phase == 'pilot', phase == 'check'
    finish = PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not checking: raise ValueError('Completed phase immutable')
    if phase == 'run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot must pass first')
    start, beat = runtime(); rows = []; receipts = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        contexts = sorted(reg['contexts'].items())
        for context, refs in contexts[:1] if pilot else contexts:
            packets, held = parent.LOADER.packets(refs, ancestor, beat); del held
            bank, spans, _ = parent.api.pool(packets)
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                paths = {s: PUBLIC/'groups'/f'{context}_{s}_seed{seed}.json' for s in sorted(packets)}
                if resume and not checking and not pilot and all(p.exists() for p in paths.values()):
                    fresh = [json.loads(p.read_text()) for p in paths.values()]
                    if any(r['registration_sha256'] != sha(REG) for r in fresh): raise ValueError('Resume freeze changed')
                else:
                    values, queries, _ = parent.predictions(context, seed, packets, origins, states)
                    q = api.np.concatenate([queries[s] for s in sorted(packets)])
                    oldrows = {}
                    for source, path in paths.items():
                        p = parent.PUBLIC/'groups'/path.name
                        if sha(p) != oldrefs[path.name]['sha256']: raise ValueError('Parent group changed')
                        oldrows[source] = json.loads(p.read_text())
                    full = api.retrieve(bank, q, selected=False, heartbeat=beat)
                    verify_full(bank, q, full, oldrows)
                    selected = api.retrieve(bank, q, selected=True, heartbeat=beat)
                    for a, b in zip(full['references'], selected['references']):
                        if b['supported'] and any(a[k] != b[k] for k in ('mean_sha256', 'std_sha256')):
                            raise ValueError('Selection changed distance normalization')
                    fresh = []
                    for source in sorted(packets):
                        lo, hi = spans[source]; take = (full['indices'] >= lo) & (full['indices'] < hi)
                        ix = full['indices'][take]-lo; _, env, y, _, _, _, _, pr = packets[source]
                        p = {a: v[ix] for a, v in values[source].items()}
                        for prefix, out in (('pooled', full), ('selected', selected)):
                            for mode in api.MODES:
                                arm = prefix+'_'+mode; p[arm] = p['origin'].copy()
                                p[arm][:, [1, 4]] = out['predictions'][mode][take]
                        for arm in api.ARMS[:5]:
                            if api.base.fingerprint(p[arm]) != oldrows[source]['prediction_sha256'][arm]:
                                raise ValueError('Inherited full-bank prediction differs')
                        if pilot: continue
                        support = selected['supported'][take]
                        scores = {a: api.base.score(y[ix], v, env[ix], pr['scale'],
                            support if a.startswith('selected_') else api.np.ones(len(ix), bool)) for a, v in p.items()}
                        for arm in api.ARMS[:5]:
                            if scores[arm] != oldrows[source]['scores'][arm]: raise ValueError('Inherited scores changed')
                        d = {prefix: api.decomposition(y[ix], out['predictions']['nearest32'][take],
                             out['probability'][take], out['severity'][take], env[ix], pr['scale'], out['supported'][take])
                             for prefix, out in (('pooled', full), ('selected', selected))}
                        row = dict(context=context, source=source, seed=seed, scope='fit', registration_sha256=sha(REG),
                            parent_group_sha256=sha(parent.PUBLIC/'groups'/paths[source].name),
                            parent_cohort_sha256=oldrows[source]['parent_cohort_sha256'], prior_predictions_exact=True,
                            references=[v for v in selected['references'] if v['query_source'] == source],
                            full_references=[v for v in full['references'] if v['query_source'] == source],
                            prediction_sha256={a: api.base.fingerprint(v) for a, v in p.items()},
                            supported_sha256=api.base.fingerprint(support), scores=scores, decomposition=d,
                            whole_pipeline_OOF=False, held_inference=False, independent_roles_read=False,
                            deployment_changed=False, result_source='fresh_run_policy_conditioned_fitting_reference')
                        if checking:
                            if row != json.loads(paths[source].read_text()): raise ValueError('Full replay differs')
                        else: once(paths[source], row)
                        fresh.append(row)
                    if pilot:
                        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-start,
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            views=4, queries=len(full['indices']), supported=int(selected['supported'].sum()),
                            minimum_selected_donors=min(r['known_donors'] for r in selected['references']),
                            engineering_pass=True, frozen_predictions_exact=True, normalization_exact=True,
                            scientific_screen_evaluated=False, registration_sha256=sha(REG)))
                        beat(state='pilot_complete'); return
                rows.extend(fresh)
                receipts.extend(dict(path=str(paths[r['source']].relative_to(ROOT)), sha256=sha(paths[r['source']])) for r in fresh)
                beat(state='verified_views' if checking else 'fit_views', views=len(rows), total=72)
        summary = api.summarize(rows, parent.previous.previous.parent.expected(reg))
        if checking:
            done = json.loads(finish.read_text())
            if done['groups'] != receipts or done['summary_sha256'] != sha(PUBLIC/'summary.json') or summary != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json', summary)
            once(finish, dict(groups=receipts, registration_sha256=sha(REG), summary_sha256=sha(PUBLIC/'summary.json'),
                pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0, held_inference=False, independent_roles_read=False, deployment_changed=False))
        beat(state='verified' if checking else 'run_complete', magnitude_screen=summary['magnitude_screen_pass'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('register', 'pilot', 'run', 'check'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    if args.action == 'register': register()
    else: execute(args.action, args.resume)
