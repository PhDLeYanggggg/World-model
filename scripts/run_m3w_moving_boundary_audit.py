"""Reuse exposed fitting records; attribute cached-model boundary errors."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_ego_frame as old
from src.evaluation import m3w_boundary_audit as method

source, rel, np = old.source, old.rel, old.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_boundary_audit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_boundary_audit_v1'


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        runtime = old.parent.old.previous.old.run.old.old.previous.teacher.lineage
        runtime.PRIVATE = PRIVATE
        start, guard, beat = runtime.runtime({}, source.read(old.CONFIG), 'audit')
        _, values = old.register(beat)
        rel.base.exact(source.read(old.PUBLIC/'verification.json')['analysis_sha256'], source.sha(old.PUBLIC/'analysis.json'))
        contexts = []
        for job in values[3]:
            d = old.inputs(values, job, guard)
            tail = old.parent.inputs(values, job, guard)
            rel.base.exact(d['ids'], tail['ids']); rel.base.exact(d['target'], tail['target'])
            old_pred = np.empty_like(d['target']); new_pred = old_pred.copy(); scales = old_pred.copy()
            folds = []
            for site in sorted(set(d['sites'])):
                held = d['sites'] == site
                state, control = old.parent.old.cached_control(dict(d, xs={'summary120': d['xs']['original120']}), job, site)
                old_pred[held] = old.method.predict(state, d['xs']['original120'][held])
                ref = source.read(old.PUBLIC/'fits'/(job['key']+'_'+str(site)+'_history121.json'))
                saved = rel.base.read_checkpoint(source.checked(ref['checkpoint']))
                rel.base.exact(saved['identity']['held_ids_hash'], rel.parent.kin.digest(d['ids'][held]))
                rel.base.exact(saved['identity']['x_hash'], rel.parent.kin.digest(d['xs']['history121']))
                new_pred[held] = old.method.predict(saved['state'], d['xs']['history121'][held])
                rel.base.exact(new_pred[held], saved['prediction']); rel.base.exact(state['target_scale'], saved['state']['target_scale'])
                scales[held] = state['target_scale']
                folds.append(dict(source=str(site), control=control, treatment=ref['checkpoint'],
                    diagnostic=method.summarize(d['target'][held], old_pred[held], new_pred[held],
                        tail['moments'][held], d['weights'][held], state['target_scale'])))
            payload = dict(x=d['xs']['original120'], ego_x=d['xs']['history121'], target=d['target'],
                moments=tail['moments'], weights=d['weights'], sites=d['sites'], query=d['query'], ids=d['ids'],
                old_prediction=old_pred, ego_prediction=new_pred, target_scale=scales)
            path = PRIVATE/(job['key']+'.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as cache:
                    for k, v in payload.items(): rel.base.exact(cache[k], v)
            else:
                np.savez_compressed(path, **payload)
            contexts.append(dict(job=job['key'], audit=d['audit'], cache=source.artifact(path),
                array_hashes={k:rel.parent.kin.digest(v) for k,v in payload.items()}, folds=folds))
            beat(state='boundary_audit_complete', job=job['key'], sources=len(folds))
            del d, tail, payload; source.gc.collect(); guard()
        source.once(PUBLIC/'analysis.json', source.stable(dict(contexts=contexts,
            result_source='fresh_run_descriptive_reanalysis_of_cached_verified_models',
            new_fits=0, new_V_or_H=False, identity_certification=False, original_rules_unchanged=True,
            deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
        source.once(PUBLIC/'execution.json', dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__': main()
