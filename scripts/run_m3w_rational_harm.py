"""Train and evaluate a single registered C1 rational-mapping repair."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from scripts import run_m3w_full_range_harm as range_run

prior = range_run.prior
from src.world_model import m3w_rational_harm as api
from src.evaluation import m3w_rational_harm_readout as evaluation

ROOT, run = prior.ROOT, prior.run
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_rational_harm_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_rational_harm_v1'
REG = PUBLIC/'registration.json'


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable receipt changed: '+str(path))
        return
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def register():
    reg, cfg, ancestor = range_run.admission()
    files = [Path(__file__).resolve(), ROOT/'src/world_model/m3w_rational_harm.py',
        ROOT/'src/evaluation/m3w_rational_harm_readout.py',
        ROOT/'tests/test_m3w_rational_harm.py', ROOT/'tests/test_m3w_rational_harm_readout.py', PUBLIC/'protocol.md']
    doc = dict(bindings={str(p.relative_to(ROOT)): run.base.sha(p) for p in files},
        parent_registration_sha256=run.base.sha(run.REG),
        parent_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=run.base.sha(prior.parent.PUBLIC/'complete.json'),
        motivation_sha256=run.base.sha(ROOT/'outputs/publication_readiness_2026_09/european_harm_jacobian_v1/summary.json'),
        range_registration_sha256=run.base.sha(range_run.REG),
        range_freeze_sha256=run.base.sha(range_run.PUBLIC/'training_freeze.json'),
        range_readout_sha256=run.base.sha(range_run.PUBLIC/'readout/complete.json'),
        contexts=reg['contexts'], seeds=reg['seeds'], fits=18, steps=2000,
        head_training=cfg['head_training'], objective=cfg['objective'],
        bootstrap_draws=cfg['bootstrap_draws'], bootstrap_seed=cfg['bootstrap_seed'],
        independent_roles_read=False, outer_development_read=False, deployment_changed=False)
    once(REG, doc)
    return reg, cfg, ancestor


def admission():
    reg, cfg, ancestor = register()
    doc = json.loads(REG.read_text())
    for rel in [*doc['bindings'], str(REG.relative_to(ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if p.returncode or p.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit protocol/code before real execution')
    return reg, cfg, ancestor


def runtime():
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def beat(**kw):
        r = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        temp = PRIVATE/'heartbeat.tmp'
        temp.write_text(json.dumps(r)+'\n')
        os.replace(temp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(r)+'\n')
        print(json.dumps(r), flush=True)
        if time.monotonic()-started > 43200:
            raise TimeoutError('12h limit; preserve checkpoints and resume without shrinking scope')
    return started, beat


def train(phase, resume):
    reg, cfg, ancestor = admission()
    _, controls = prior.parent.models(reg)
    _, anchors = run.ancestor_eval.models()
    finish = PUBLIC/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if phase == 'train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real resume/protection pilot required')
    started, beat = runtime()
    refs = []

    def storage():
        if shutil.disk_usage(ROOT).free < 10*2**30 or sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > 256*2**20:
            raise OSError('Preserve10GiB reserve and256MiB checkpoint cap')
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        storage()
        contexts = sorted(reg['contexts'])[:1] if phase == 'pilot' else sorted(reg['contexts'])
        for context in contexts:
            fit, _ = run.parent.prior.packets(reg['contexts'][context], ancestor, beat)
            for seed in (reg['seeds'][:1] if phase == 'pilot' else reg['seeds']):
                ref = anchors[context, seed, 'shared_utility']['checkpoint']
                anchor = api.core.read_checkpoint(ROOT/ref['path'])
                identity = dict(context=context, seed=seed, partitions={i['packet']['identity']['source']: i['partition'] for i in reg['contexts'][context]})
                key = context+'_head'+str(seed)
                path = PRIVATE/('pilot' if phase == 'pilot' else 'heads')/(key+'.pt.gz')
                kw = dict(anchor=anchor, anchor_sha256=ref['sha256'], arm='isolated_risk',
                    settings=cfg['head_training'], objective_config=cfg['objective'], identity=identity,
                    experiment_sha256=run.base.sha(REG), heartbeat=beat, guard=storage)
                if phase == 'pilot' and not path.exists():
                    api.fit(fit, path=path, stop_at=50, **kw)
                state = api.fit(fit, path=path, resume=(phase == 'pilot') or (resume and path.exists()),
                                stop_at=100 if phase == 'pilot' else None, **kw)
                control = api.core.read_checkpoint(ROOT/controls[context, seed, 'isolated_risk']['checkpoint']['path'])
                for metric in ('loss', 'primary_loss', 'regret', 'mean_soft_action'):
                    api.core.exact(state['trace'][0][metric], control['trace'][0][metric])
                if phase == 'pilot':
                    direct = api.fit(fit, path=PRIVATE/'pilot/direct.pt.gz', resume=resume,
                                     stop_at=100, **kw)
                    for k in state:
                        if k != 'seconds':
                            api.core.exact(state[k], direct[k])
                    for source, args in fit.items():
                        x, env = args[0][:256], args[1][:256]
                        pred = api.predict(state, source, x, env)
                        truth = api.previous.predict(anchor, source, x, env)
                        api.np.testing.assert_array_equal(pred[:, [0, 2, 3]], truth[:, [0, 2, 3]])
                    elapsed = time.monotonic()-started
                    once(finish, dict(pid=os.getpid(), seconds=elapsed, actual_updates=200,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        engineering_pass=True, exact_resume=True, protected_outputs_exact=True,
                        estimated_full_seconds=elapsed*180*1.5, registration_sha256=run.base.sha(REG)))
                else:
                    control = api.core.read_checkpoint(ROOT/controls[context, seed, 'isolated_risk']['checkpoint']['path'])
                    api.assert_control_match(state, control)
                    doc = dict(context=context, seed=seed, identity=identity, steps=2000,
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=run.base.sha(path)),
                        registration_sha256=run.base.sha(REG), matched_initialization_sampling=True,
                        protected_outputs_exact=True, final_trace=state['trace'][-1],
                        result_source='fresh_run_rational_harm_continuation', new_forecaster=False,
                        trainable_parameters=state['trainable_parameters'], total_parameters=state['total_parameters'])
                    p = PUBLIC/'fits'/(key+'.json')
                    once(p, doc)
                    refs.append(dict(path=str(p.relative_to(ROOT)), sha256=run.base.sha(p)))
                    beat(state='fit_complete', fits=len(refs), total=18)
        if phase == 'train':
            if len(refs) != 18:
                raise ValueError('Complete18 required')
            once(finish, dict(registration_sha256=run.base.sha(REG), fits=refs, fresh_fits=18, optimizer_steps=36000,
                pid=os.getpid(), seconds=time.monotonic()-started, held_scored=False,
                independent_roles_read=False, deployment_changed=False))
        beat(state=phase+'_complete', fits=len(refs))


def models(reg):
    frozen = json.loads((PUBLIC/'training_freeze.json').read_text())
    if (frozen['registration_sha256'] != run.base.sha(REG) or frozen['fresh_fits'] != 18
            or frozen['optimizer_steps'] != 36000 or frozen['held_scored']):
        raise ValueError('All18 finals frozen before readout required')
    docs = {}
    for ref in frozen['fits']:
        p = ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to((PUBLIC/'fits').resolve()) or run.base.sha(p) != ref['sha256']:
            raise ValueError('Changed final receipt')
        d = json.loads(p.read_text())
        ck = ROOT/d['checkpoint']['path']
        if ck.is_symlink() or not ck.resolve().is_relative_to((PRIVATE/'heads').resolve()) or run.base.sha(ck) != d['checkpoint']['sha256']:
            raise ValueError('Changed checkpoint')
        if d['registration_sha256'] != run.base.sha(REG) or d['steps'] != 2000:
            raise ValueError('Changed fit binding')
        docs[d['context'], d['seed']] = d
    if set(docs) != {(c, s) for c in reg['contexts'] for s in reg['seeds']}:
        raise ValueError('Complete unique fit roster required')
    return frozen, docs


def range_rows():
    complete = json.loads((range_run.PUBLIC/'readout/complete.json').read_text())
    rows = {}
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if path.is_symlink() or not path.resolve().is_relative_to((range_run.PUBLIC/'readout/groups').resolve()) or run.base.sha(path) != ref['sha256']:
            raise ValueError('Changed range comparator receipt')
        row = json.loads(path.read_text())
        key = row['scope'], row['context'], row['source'], row['seed']
        if key in rows:
            raise ValueError('Duplicate range comparator identity')
        rows[key] = row
    if len(rows) != 144:
        raise ValueError('Complete144 comparator views required')
    return rows


def evaluate(resume=False, check=False):
    reg, cfg, ancestor = admission()
    _, docs = models(reg)
    _, controls = prior.parent.models(reg)
    _, anchors = run.ancestor_eval.models()
    _, range_models = range_run.models(reg)
    locals_ = run.parent.previous_eval.frozen_models()
    old = range_rows()
    out = PUBLIC/'readout'
    if check:
        complete = json.loads((out/'complete.json').read_text())
        rows = []
        for ref in complete['groups']:
            p = ROOT/ref['path']
            if p.is_symlink() or not p.resolve().is_relative_to((out/'groups').resolve()) or run.base.sha(p) != ref['sha256']:
                raise ValueError('Changed readout receipt')
            r = json.loads(p.read_text())
            original = old[r['scope'], r['context'], r['source'], r['seed']]
            if r['registration_sha256'] != run.base.sha(REG):
                raise ValueError('Changed readout binding')
            for a in evaluation.CONTROLS:
                api.core.exact(r['result']['policies'][a], original['result']['policies'][a])
                api.core.exact(r['result']['scores'][a], original['result']['scores'][a])
                api.core.exact(r['prediction_hashes'][a], original['prediction_hashes'][a])
            evaluation.validate(r['result'])
            rows.append(r)
        if complete['summary_sha256'] != run.base.sha(out/'summary.json') or evaluation.summarize(rows, old, cfg) != json.loads((out/'summary.json').read_text()):
            raise ValueError('Summary replay failed')
        for (c, s), doc in docs.items():
            state = api.core.read_checkpoint(ROOT/doc['checkpoint']['path'])
            control = api.core.read_checkpoint(ROOT/controls[c, s, 'isolated_risk']['checkpoint']['path'])
            api.assert_control_match(state, control)
        print('Verified18 matched final checkpoints,144 views, seven old policies and risk bounds')
        return
    if (out/'complete.json').exists():
        raise ValueError('Completed evaluation immutable')
    started, beat = runtime()
    rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = run.parent.prior.packets(items, ancestor, beat)
            for seed in reg['seeds']:
                state = api.core.read_checkpoint(ROOT/docs[context, seed]['checkpoint']['path'])
                range_state = api.core.read_checkpoint(ROOT/range_models[context, seed]['checkpoint']['path'])
                continued = [api.core.read_checkpoint(ROOT/controls[context, seed, a]['checkpoint']['path']) for a in prior.api.ARMS]
                api.assert_control_match(state, continued[-1])
                anchor, risk = [api.core.read_checkpoint(ROOT/anchors[context, seed, a]['checkpoint']['path']) for a in ('shared_utility', 'shared_risk')]
                for source in sorted(fit):
                    local = api.core.read_checkpoint(ROOT/locals_[context, seed, 'local', source]['checkpoint']['path'])
                    for scope, args in (('fit', fit[source]), ('held', held[source])):
                        pth = out/'groups'/(scope+'_'+context+'_'+source+'_seed'+str(seed)+'.json')
                        original = old[scope, context, source, seed]
                        if pth.exists() and not resume:
                            raise ValueError('Explicit resume required')
                        if pth.exists():
                            row = json.loads(pth.read_text())
                            if row['registration_sha256'] != run.base.sha(REG):
                                raise ValueError('Interrupted binding changed')
                        else:
                            x, env = args[:2]
                            pp = dict(local=prior.api.previous.old.predict(local, x, env)[0],
                                anchor=api.previous.predict(anchor, source, x, env),
                                old_risk=api.previous.predict(risk, source, x, env),
                                full_range=range_run.api.predict(range_state, source, x, env),
                                rational=api.predict(state, source, x, env))
                            pp.update({s['arm']: prior.api.predict(s, source, x, env) for s in continued})
                            api.np.testing.assert_array_equal(pp['rational'], api.predict(state, source, x, env))
                            hashes = {a: api.previous.old.parent.forest.fingerprint(p) for a, p in pp.items()}
                            result = evaluation.evaluate(args, pp, api.previous.prior.eligible(args))
                            evaluation.validate(result)
                            for a in evaluation.CONTROLS:
                                api.core.exact(hashes[a], original['prediction_hashes'][a])
                                api.core.exact(result['policies'][a], original['result']['policies'][a])
                                api.core.exact(result['scores'][a], original['result']['scores'][a])
                            row = dict(registration_sha256=run.base.sha(REG), scope=scope, context=context,
                                source=source, seed=seed, prediction_hashes=hashes, result=result,
                                result_source='fresh_run_frozen_final_inference', independent_roles_read=False,
                                deployment_changed=False, new_forecaster=False)
                            once(pth, row)
                        rows.append(row)
                        refs.append(dict(path=str(pth.relative_to(ROOT)), sha256=run.base.sha(pth)))
                        beat(state='view_complete', views=len(rows), total=144)
        summary = evaluation.summarize(rows, old, cfg)
        once(out/'summary.json', summary)
        once(out/'complete.json', dict(registration_sha256=run.base.sha(REG), groups=refs,
            summary_sha256=run.base.sha(out/'summary.json'), seconds=time.monotonic()-started,
            pid=os.getpid(), independent_roles_read=False, deployment_changed=False))
        beat(state='readout_complete', advance=summary['advance'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'pilot', 'train', 'eval', 'check'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action in ('pilot', 'train'):
        train(args.action, args.resume)
    else:
        evaluate(args.resume, args.action == 'check')
