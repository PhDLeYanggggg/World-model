"""Continue frozen risk heads, then evaluate a single fixed-budget checkpoint."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_neural_transport as old
from src.world_model import m3w_risk_convergence as method

source, rel, np = old.source, old.rel, old.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_risk_convergence_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_risk_convergence_v1'
CONFIG = ROOT/'configs/m3w_moving_risk_convergence_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
         ROOT/'src/world_model/m3w_risk_convergence.py',
         ROOT/'tests/test_m3w_risk_convergence.py')
ARMS = ('shared64', 'track32', 'short_plain', 'short_transport', 'long_plain', 'long_transport')


def register():
    old.register()
    cfg = source.read(CONFIG)
    if ({k: cfg[k] for k in method.BUDGET} != method.BUDGET
            or cfg['primary'] != 'continued_transport_vs_shared64'
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or cfg['minimum_joint_cost_reduction'] != .01 or cfg['minimum_passing_contexts'] != 2
            or any(cfg[k] for k in ('threshold_search', 'held_epoch_selection', 'identity_certification',
                                   'new_H_readout', 'deployment_changed', 'submission_ready',
                                   'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed budget and original rules required')
    previous = source.read(old.PUBLIC/'analysis.json')
    rel.base.exact(source.sha(old.PUBLIC/'analysis.json'),
                   source.read(old.PUBLIC/'verification.json')['analysis_sha256'])
    source.once(REG, source.stable(dict(
        config=cfg, files={str(f.relative_to(ROOT)): source.sha(f) for f in (*FILES, *old.FILES)},
        parent_registration=source.artifact(old.REG), parent_result=source.artifact(old.PUBLIC/'analysis.json'),
        role='exposed_own_B_source_held_optimization_diagnostic',
        original_rules_unchanged=True, new_V_or_H=False, identity_certification=False)))
    return cfg, previous


def initial(d, context, f):
    state = rel.base.read_checkpoint(source.checked(f['neural_checkpoint']))
    rel.base.exact(state['identity'], old.identity(d, context['job'], f['source']))
    rel.base.exact(state['step'], f['steps'])
    return state


def binding(d, c, f):
    return dict(registration_sha256=source.sha(REG), parent=f['neural_checkpoint'],
                causal_input_hash=method.parent.digest(d['x']),
                row_ids_hash=method.parent.digest(d['ids']), job=c['job'], source=f['source'])


def checkpoint(c, f, directory='fits'):
    return PRIVATE/directory/(c['job']+'_'+f['source']+'.pt.gz')


def train_fold(d, c, f, guard, beat, *, resume, verify=False):
    start = initial(d, c, f)
    path = checkpoint(c, f, 'verify' if verify else 'fits')
    state = method.continue_fit(d, f['source'], start, binding(d, c, f), path,
                                resume=resume, heartbeat=beat, guard=guard)
    if not state['complete']:
        raise ValueError('Full continuation required')
    if verify:
        method.exact_state(rel.base.read_checkpoint(checkpoint(c, f)), state)
    else:
        source.once(PUBLIC/'fits'/(path.stem+'.json'), source.stable(dict(
            source=f['source'], job=c['job'], checkpoint=source.artifact(path),
            fit_rows=state['fit_rows'], parameters=state['parameters'],
            start_step=state['start_step'], final_step=state['step'], epochs=state['epoch'],
            extra_updates=state['step']-state['start_step'], minimum_visits=int(state['visits'].min()),
            extra_trace=state['extra_trace'], continuation_seconds=state['continuation_seconds'],
            fitting_diagnosis=method.fitting_diagnosis(state), result_source='fresh_run')))
    return state


def summarize(folds):
    keys = ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')
    means = {a: {k: float(np.mean([f['metrics'][a][k] for f in folds])) for k in keys} for a in ARMS}
    ref, long, plain = means['shared64'], means['long_transport'], means['long_plain']
    gain = 1-long['total_joint_cost']/ref['total_joint_cost'] if ref['total_joint_cost']>0 else None
    return dict(source_macro=means, cost_reduction=gain, passes_information_screen=bool(
        gain is not None and gain>=.01 and long['false_safe_cost']<=ref['false_safe_cost']
        and long['total_joint_cost']<plain['total_joint_cost']), safety_certificate=False)


def evaluate_fold(d, c, f):
    start = initial(d, c, f)
    state = rel.base.read_checkpoint(checkpoint(c, f))
    rel.base.exact(state['identity'], binding(d, c, f))
    if not state['complete']:
        raise ValueError('No intermediate held-out readout')
    reference = rel.base.read_checkpoint(source.checked(f['reference_checkpoint']))['result']
    at = d['sites'] == f['source']
    pos = reference['donor_positions']
    if (d['sites'][pos] == f['source']).any():
        raise ValueError('Query source in reference bank')
    rel.base.exact(d['ids'][pos], reference['donor_ids'])
    outputs = dict(shared64=d['old_prediction'][at], track32=reference['prediction'])
    fitting = {}
    metric = old.old.method.base.metrics
    for family, model in (('short', start), ('long', state)):
        pred = method.parent.predict(model, d['x'])
        outputs[family+'_plain'] = pred[at]
        outputs[family+'_transport'] = method.parent.transport(
            pred[at], pred[pos], d['target'][pos], d['weights'][pos])
        fitting[family] = metric(pred[~at], d['target'][~at], d['weights'][~at], model['preprocess']['target_scale'])
    for arm in ('plain', 'transport'):
        rel.base.exact(method.parent.digest(outputs['short_'+arm]), f['prediction_hashes']['neural_'+arm])
    metrics = {k: metric(p, d['target'][at], d['weights'][at], state['preprocess']['target_scale'])
               for k, p in outputs.items()}
    for a, b in (('shared64', 'shared64'), ('track32', 'track32'),
                 ('short_plain', 'neural_plain'), ('short_transport', 'neural_transport')):
        rel.base.exact(metrics[a], f['metrics'][b])
    return source.stable(dict(
        source=f['source'], rows=int(at.sum()), fit_rows=int((~at).sum()),
        checkpoint=source.artifact(checkpoint(c, f)), parent_checkpoint=f['neural_checkpoint'],
        reference_checkpoint=f['reference_checkpoint'], parameters=state['parameters'],
        start_step=state['start_step'], final_step=state['step'], epochs=state['epoch'],
        extra_updates=state['step']-state['start_step'], fitting_diagnosis=method.fitting_diagnosis(state),
        fitting_metrics=fitting, metrics=metrics,
        prediction_hashes={k: method.parent.digest(p) for k, p in outputs.items()},
        boundary=old.boundary(d['target'][at], outputs['shared64'], outputs['long_transport'],
                             d['moments'][at], d['weights'][at], state['preprocess']['target_scale']),
        result_source='fresh_run_continuation_and_final_development_evaluation', controls_source='cached_verified'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'fit', 'verify', 'eval', 'replay'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime = old.old.previous.audit.old.parent.old.previous.old.run.old.old.previous.teacher.lineage
        runtime.PRIVATE = PRIVATE
        begin, guard, beat = runtime.runtime({}, source.read(CONFIG), args.phase)
        before = {str(p): source.sha(p) for p in PUBLIC.rglob('*') if p.is_file()} if args.phase=='replay' else None
        cfg, previous = register()
        if args.phase=='register':
            return
        for file in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(file.relative_to(ROOT))],
                                             cwd=ROOT, capture_output=True, check=True).stdout
            rel.base.exact(source.hashlib.sha256(content).hexdigest(), source.sha(file))
        receipt = (PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase immutable; do not restart it')
        if args.phase=='pilot':
            c = previous['contexts'][-1]
            f = c['folds'][0]
            d, _ = old.load(c)
            start, ident = initial(d, c, f), binding(d, c, f)
            path = PRIVATE/'pilot/resumed.pt.gz'
            if not path.exists():
                method.continue_fit(d, f['source'], start, ident, path, stop_extra_steps=128, heartbeat=beat, guard=guard)
            b = method.continue_fit(d, f['source'], start, ident, path, resume=True, stop_extra_steps=256, heartbeat=beat, guard=guard)
            a = method.continue_fit(d, f['source'], start, ident, PRIVATE/'pilot/full.pt.gz',
                                    resume=args.resume, stop_extra_steps=256, heartbeat=beat, guard=guard)
            method.exact_state(a, b)
            source.once(PUBLIC/'pilot.json', dict(real256update_resume_exact=True, fit_rows=a['fit_rows'],
                seconds=a['continuation_seconds'], rough8fold_seconds=a['continuation_seconds']*16*8,
                checkpoint=source.artifact(path)))
        elif args.phase in ('fit', 'verify'):
            if not source.read(PUBLIC/'pilot.json')['real256update_resume_exact']:
                raise ValueError('Real recovery pilot required')
            total = 0
            for c in previous['contexts']:
                d, _ = old.load(c)
                for f in c['folds']:
                    state = train_fold(d, c, f, guard, beat, resume=args.resume, verify=args.phase=='verify')
                    total += state['step']-state['start_step']
                    beat(state='fold_complete', job=c['job'], source=f['source'])
            source.once(PUBLIC/(args.phase+'_completion.json'), dict(folds=8, additional_updates=total,
                all_fitting_rows_complete_epochs=True, source_held_evaluated=False,
                all8_continuations_exact=args.phase=='verify'))
        else:
            if not source.read(PUBLIC/'verify_completion.json')['all8_continuations_exact']:
                raise ValueError('Finish and verify all fits before readout')
            def forbidden(*a, **kw):
                raise RuntimeError('Fitting forbidden during evaluation/replay')
            method.continue_fit = forbidden
            method.parent.fit = forbidden
            contexts = []
            for c in previous['contexts']:
                d, _ = old.load(c)
                folds = [evaluate_fold(d, c, f) for f in c['folds']]
                contexts.append(dict(job=c['job'], folds=folds, summary=summarize(folds)))
            result = source.stable(dict(contexts=contexts,
                passing_contexts=sum(c['summary']['passes_information_screen'] for c in contexts),
                original_rules_unchanged=True, identity_certification=False, new_V_or_H=False,
                original24group_primary='not_estimable_not_passed', deployment_changed=False,
                independent_confirmation=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
            if args.phase=='replay':
                rel.base.exact(source.read(PUBLIC/'analysis.json'), result)
            else:
                source.once(PUBLIC/'analysis.json', result)
        unchanged = None
        if before is not None:
            rel.base.exact(before, {str(p): source.sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
            unchanged = len(before)
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-begin,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            unchanged_public_files=unchanged))
        beat(state='complete')


if __name__=='__main__':
    main()
