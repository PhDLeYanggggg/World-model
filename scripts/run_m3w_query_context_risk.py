"""Registered same-query context experiment, using only original TRAIN packets."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')

from scripts import run_m3w_rational_harm as parent
from src.world_model import m3w_query_context_risk as api
from src.evaluation import m3w_query_context_readout as evaluation

ROOT = parent.ROOT
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_query_context_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_query_context_v1'
REG = PUBLIC/'registration.json'
sha, once = parent.run.base.sha, parent.once


def register():
    reg, cfg, ancestor = parent.admission()
    files = [Path(__file__).resolve(), PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_query_context_risk.py', ROOT/'src/evaluation/m3w_query_context_readout.py',
        ROOT/'tests/test_m3w_query_context_risk.py', ROOT/'tests/test_m3w_query_context_readout.py']
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in files},
        parent_registration_sha256=sha(parent.REG), parent_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=sha(parent.PUBLIC/'readout/complete.json'),
        preceding_failure_sha256=sha(ROOT/'outputs/publication_readiness_2026_09/european_policy_conditional_v1/summary.json'),
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], arms=list(api.ARMS), fits=36,
        head_training=cfg['head_training'], objective=cfg['objective'],
        bootstrap_draws=cfg['bootstrap_draws'], bootstrap_seed=cfg['bootstrap_seed'],
        independent_roles_read=False, outer_development_read=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    reg, cfg, ancestor = register(); doc = json.loads(REG.read_text())
    for rel in [*doc['bindings'], str(REG.relative_to(ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if p.returncode or p.stdout != (ROOT/rel).read_bytes(): raise ValueError('Commit experiment before real execution')
    return reg, cfg, ancestor


def runtime():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    def beat(**kw):
        d = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(d)+'\n'); os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(d)+'\n')
        print(json.dumps(d), flush=True)
        if time.monotonic()-started > 43200: raise TimeoutError('12h boundary; resume unchanged experiment')
    return started, beat


def storage():
    used = sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    if shutil.disk_usage(ROOT).free < 10*2**30 or used > 256*2**20:
        raise OSError('Preserve10GiB reserve and256MiB checkpoint cap')


def train(phase, resume):
    reg, cfg, ancestor = admission(); _, anchors = parent.run.ancestor_eval.models()
    finish = PUBLIC/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if phase == 'train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot/resume check required')
    began, beat = runtime(); refs = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); storage()
        contexts = sorted(reg['contexts'])[:1] if phase == 'pilot' else sorted(reg['contexts'])
        for context in contexts:
            fit, held = parent.run.parent.prior.packets(reg['contexts'][context], ancestor, beat)
            del held
            for seed in (reg['seeds'][:1] if phase == 'pilot' else reg['seeds']):
                ref = anchors[context, seed, 'shared_utility']['checkpoint']
                anchor = api.core.read_checkpoint(ROOT/ref['path']); states = []
                identity = dict(context=context, seed=seed,
                    partitions={i['packet']['identity']['source']: i['partition'] for i in reg['contexts'][context]})
                for arm in api.ARMS:
                    key = context+'_head'+str(seed)+'_'+arm
                    path = PRIVATE/('pilot' if phase == 'pilot' else 'heads')/(key+'.pt.gz')
                    kw = dict(anchor=anchor, anchor_sha256=ref['sha256'], arm=arm,
                        settings=cfg['head_training'], objective_config=cfg['objective'], identity=identity,
                        experiment_sha256=sha(REG), path=path, heartbeat=beat, guard=storage)
                    if phase == 'pilot' and arm == 'peer' and not path.exists(): api.fit(fit, stop_at=50, **kw)
                    state = api.fit(fit, resume=path.exists() and (resume or phase == 'pilot'),
                        stop_at=100 if phase == 'pilot' else None, **kw)
                    states.append(state)
                    if phase == 'train':
                        out = PUBLIC/'fits'/(key+'.json')
                        once(out, dict(context=context, seed=seed, arm=arm, identity=identity,
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            registration_sha256=sha(REG), steps=state['step'], seconds=state['seconds'],
                            final_trace=state['trace'][-1], initial_trace=state['trace'][0],
                            context_stats=state['context_stats'], trainable_parameters=state['trainable_parameters'],
                            total_parameters=state['total_parameters'], result_source='fresh_run_real_Torch_risk_training',
                            new_forecaster=False, held_scored=False, independent_roles_read=False))
                        refs.append(dict(path=str(out.relative_to(ROOT)), sha256=sha(out)))
                        beat(state='fit_complete', fits=len(refs), total=36)
                api.assert_matched(*states)
                if phase == 'pilot':
                    direct = api.fit(fit, **{**kw, 'path': PRIVATE/'pilot/direct.pt.gz'},
                                    resume=resume, stop_at=100)
                    for k in direct:
                        if k != 'seconds': api.core.exact(direct[k], states[-1][k])
                    for source, args in fit.items():
                        for state in states:
                            pred = api.predict(state, source, args[0], args[1], args[5], args[6])
                            teacher = api.parent.previous.predict(anchor, source, args[0], args[1])
                            api.np.testing.assert_array_equal(pred[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
                    elapsed = time.monotonic()-began
                    once(finish, dict(pid=os.getpid(), seconds=elapsed, actual_updates=300,
                        architecture=platform.machine(), torch=api.torch.__version__, numpy=api.np.__version__,
                        cpu_threads=4, interop_threads=1, workers=0,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        exact_resume=True, protection_exact=True, matched_draws=True, engineering_pass=True,
                        context_stats=states[-1]['context_stats'],
                        estimated_full_fit_seconds=sum(s['seconds'] for s in states)*20*18*1.5,
                        registration_sha256=sha(REG), scientific_lift_established=False))
        if phase == 'train':
            if len(refs) != 36: raise ValueError('Complete36 fixed fits required')
            once(finish, dict(registration_sha256=sha(REG), fits=refs, fresh_fits=36, optimizer_steps=72000,
                seconds=time.monotonic()-began, pid=os.getpid(),
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                held_scored=False, independent_roles_read=False, deployment_changed=False))
        beat(state=phase+'_complete', fits=len(refs))


def models(reg):
    frozen = json.loads((PUBLIC/'training_freeze.json').read_text())
    if frozen['registration_sha256'] != sha(REG) or frozen['fresh_fits'] != 36 or frozen['held_scored']:
        raise ValueError('All finals frozen before readout required')
    docs = {}
    for ref in frozen['fits']:
        p = ROOT/ref['path']; d = json.loads(p.read_text()); ck = ROOT/d['checkpoint']['path']
        if (p.is_symlink() or not p.resolve().is_relative_to((PUBLIC/'fits').resolve()) or sha(p) != ref['sha256']
                or ck.is_symlink() or not ck.resolve().is_relative_to((PRIVATE/'heads').resolve())
                or sha(ck) != d['checkpoint']['sha256'] or d['registration_sha256'] != sha(REG) or d['steps'] != 2000):
            raise ValueError('Changed fit receipt or checkpoint')
        key = d['context'], d['seed'], d['arm']
        if key in docs: raise ValueError('Duplicate fit')
        docs[key] = d
    if set(docs) != {(c, s, a) for c in reg['contexts'] for s in reg['seeds'] for a in api.ARMS}:
        raise ValueError('Incomplete roster')
    return frozen, docs


def expected(reg):
    return {(c, i['packet']['identity']['source'], s) for c, items in reg['contexts'].items()
            for i in items for s in reg['seeds']}


def evaluate(check, resume):
    reg, cfg, ancestor = admission(); _, docs = models(reg)
    _, rational = parent.models(reg); _, controls = parent.prior.parent.models(reg)
    _, anchors = parent.run.ancestor_eval.models(); _, ranges = parent.range_run.models(reg)
    locals_ = parent.run.parent.previous_eval.frozen_models()
    old_complete = json.loads((parent.PUBLIC/'readout/complete.json').read_text())
    old = {}
    for ref in old_complete['groups']:
        p = ROOT/ref['path']
        if sha(p) != ref['sha256']: raise ValueError('Changed comparator readout')
        row = json.loads(p.read_text())
        old[row['scope'], row['context'], row['source'], row['seed']] = row
    if not check and (PUBLIC/'complete.json').exists(): raise ValueError('Completed readout immutable')
    began, beat = runtime(); rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = parent.run.parent.prior.packets(items, ancestor, beat); del fit
            for seed in reg['seeds']:
                states = [api.core.read_checkpoint(ROOT/docs[context, seed, a]['checkpoint']['path']) for a in api.ARMS]
                api.assert_matched(*states)
                rat = api.core.read_checkpoint(ROOT/rational[context, seed]['checkpoint']['path'])
                ran = api.core.read_checkpoint(ROOT/ranges[context, seed]['checkpoint']['path'])
                continued = [api.core.read_checkpoint(ROOT/controls[context, seed, a]['checkpoint']['path']) for a in parent.prior.api.ARMS]
                anchor, risk = [api.core.read_checkpoint(ROOT/anchors[context, seed, a]['checkpoint']['path']) for a in ('shared_utility', 'shared_risk')]
                for source, args in sorted(held.items()):
                    local = api.core.read_checkpoint(ROOT/locals_[context, seed, 'local', source]['checkpoint']['path'])
                    original = old['held', context, source, seed]
                    x, env = args[:2]
                    pp = dict(local=parent.prior.api.previous.old.predict(local, x, env)[0],
                        anchor=api.parent.previous.predict(anchor, source, x, env),
                        old_risk=api.parent.previous.predict(risk, source, x, env),
                        full_range=parent.range_run.api.predict(ran, source, x, env),
                        rational=api.parent.predict(rat, source, x, env))
                    pp.update({s['arm']: parent.prior.api.predict(s, source, x, env) for s in continued})
                    pp.update({s['arm']: api.predict(s, source, x, env, args[5], args[6]) for s in states})
                    hashes = {a: api.fingerprint(p) for a, p in pp.items()}
                    result = evaluation.evaluate(args, pp, api.parent.previous.prior.eligible(args)); evaluation.validate(result)
                    for a in parent.evaluation.ARMS:
                        api.core.exact(hashes[a], original['prediction_hashes'][a])
                        api.core.exact(result['policies'][a], original['result']['policies'][a])
                        api.core.exact(result['scores'][a], original['result']['scores'][a])
                    _, _, context_stats = api.context_inputs(x, args[5], args[6], args[-1])
                    row = dict(context=context, source=source, seed=seed, result=result,
                        prediction_hashes=hashes, context_stats=context_stats, registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_run_frozen_TRAIN_internal_held_readout', independent_roles_read=False)
                    out = PUBLIC/'readout'/(context+'_'+source+'_seed'+str(seed)+'.json')
                    if check:
                        if json.loads(out.read_text()) != row: raise ValueError('Full inference/readout replay failed')
                    else:
                        if out.exists() and not resume: raise ValueError('Explicit resume required')
                        once(out, row)
                    rows.append(row); refs.append(dict(path=str(out.relative_to(ROOT)), sha256=sha(out)))
                    beat(state='verified_view' if check else 'readout_view', views=len(rows), total=72)
        summary = evaluation.summarize(rows, expected(reg), cfg)
        if check:
            saved = json.loads((PUBLIC/'complete.json').read_text())
            if saved['readouts'] != refs or saved['summary_sha256'] != sha(PUBLIC/'summary.json') or json.loads((PUBLIC/'summary.json').read_text()) != summary:
                raise ValueError('Aggregate/bootstrap replay failed')
        else:
            once(PUBLIC/'summary.json', summary)
            once(PUBLIC/'complete.json', dict(readouts=refs, summary_sha256=sha(PUBLIC/'summary.json'),
                registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                pid=os.getpid(), seconds=time.monotonic()-began, old_eight_controls_replayed=True,
                independent_roles_read=False, deployment_changed=False))
        beat(state='verified' if check else 'readout_complete', advance=summary['advance'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'pilot', 'train', 'eval', 'check'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if a.action == 'register': register()
    elif a.action in ('pilot', 'train'): train(a.action, a.resume)
    else: evaluate(a.action == 'check', a.resume)
