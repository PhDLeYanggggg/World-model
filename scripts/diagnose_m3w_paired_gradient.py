"""Frozen paired-loss gradient audit; only own-B, no new held predictions."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_paired_decision as run
from src.evaluation import m3w_paired_gradient as method

source, api = run.source, run.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_paired_gradient_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_paired_gradient_v1'


def report(result):
    f = lambda v: 'NA' if v is None else f'{v:.6g}'
    text = ['# Paired-Loss Gradient Audit', '',
        'All nine frozen models, complete own-B fitting populations and exact',
        'saved query-frequency weights. No new V/H readout or saved model update.', '',
        '| Context | Arm | Risk/rank0 cosine | Risk/rank1999 cosine | Base/rank1999 cosine | Extra-risk norm / anchor |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['rows']:
        text.append('| '+r['job']+' | '+r['arm']+' | '+' | '.join(f(v) for v in (
            r['relations']['risk_rank0']['cosine'], r['relations']['risk_rank1999']['cosine'],
            r['relations']['base_rank1999']['cosine'],
            r['relations']['risk_rank1999']['auxiliary_anchor_norm_ratio']))+' |')
    text += ['', '## Disposable Saved-AdamW Steps', '',
        '| Context | Variant | Mean risk MSE | Joint decision cost | False-safe | False-reject |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['rows']:
        for name, v in r['virtual_steps'].items():
            a = v['joint_attribution']['total']
            text.append(f"| {r['job']} | {name} | {v['fitting']['mean_MSE']:.9g} | "
                f"{a['total_joint_cost']:.9g} | {a['false_safe_count']} | {a['false_reject_count']} |")
    text += ['', f"Risk/paired conflict at both offsets: {result['rank_conflicting_contexts']}/3.",
        f"Protected step lowers MSE without higher joint cost: {result['protected_better_contexts']}/3.",
        f"Prospective training follow-up supported: **{result['protected_training_followup_supported']}**.", '',
        'These are final-model local derivatives and disposable full-fitting steps,',
        'not the full training trajectory, held generalization or an empirical safety',
        'certificate. AdamW momentum and clipping are retained. Unknown-label bounds,',
        'original 2% rules and support remain unchanged. No identity exemption.',
        'A negative result rejects this particular conflict repair, not all models.', '',
        'Original 24-group primary remains NA; independent confirmation not_run.',
        'Native obs8/pred12 image coordinates, not metric/seconds/true3D/foundation/',
        'human gold. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Metrics](analysis.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--replay', action='store_true', help='Verify a completed audit without replacing receipts')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'paired_gradient')
        _, values = run.register(beat)
        api.base.exact(source.read(run.PUBLIC/'verification.json')['analysis_sha256'],
            source.sha(run.PUBLIC/'analysis.json'))
        files = (Path(__file__), ROOT/'src/evaluation/m3w_paired_gradient.py',
            ROOT/'tests/test_m3w_paired_gradient.py', PUBLIC/'protocol.md')
        current_registration = dict(files={str(f.relative_to(ROOT)): source.sha(f) for f in files},
            parent=source.artifact(run.PUBLIC/'verification.json'), own_B_only=True,
            full_fitting_query_weights=True, pair_offsets=[0, 1999], disposable_updates=12,
            followup_rule='at_least2_rank_contexts_conflict_both_offsets_and2_protected_steps_better_MSE_nonworse_joint_cost',
            no_new_V_or_H=True, original_rules_unchanged=True, identity_certification=False)
        original_registration = source.read(PUBLIC/'registration.json')
        # A test-only fixture correction and this replay wrapper do not alter
        # the frozen evaluator, protocol, checkpoint lineage or analysis.
        runner, test = str(files[0].relative_to(ROOT)), str(files[2].relative_to(ROOT))
        compared = dict(current_registration, files=dict(current_registration['files']))
        for name in (runner, test):
            compared['files'][name] = original_registration['files'][name]
        api.base.exact(compared, original_registration)
        revision = dict(original_registration=source.artifact(PUBLIC/'registration.json'),
            current_registration=current_registration,
            changed_files=[runner, test], scientific_evaluator_unchanged=True,
            reason='Correct mutated-mask test to assert rejection; add read-only completed replay')
        if args.replay:
            api.base.exact(source.read(PUBLIC/'validation_revision.json'), revision)
            before = {str(f): source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()}
        else:
            source.once(PUBLIC/'validation_revision.json', revision)
        if args.verify and not (PUBLIC/'analysis.json').exists():
            raise ValueError('Initial diagnostic required before replay')
        rows = []
        for job in values[3]:
            data, origin = run.inputs(values, job)
            for arm in api.ARMS:
                receipt = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                ref = receipt['checkpoint']; state = api.base.read_checkpoint(source.checked(ref))
                losses, grads, names = method.full(state, data, guard)
                source.np.testing.assert_allclose(2*losses['risk'], receipt['final_fitting']['mean_MSE'], rtol=1e-6, atol=1e-8)
                ids = [i for i, name in enumerate(names) if name.startswith('risk.')]
                select = lambda g: [g[i] for i in ids]
                base = [sum(g) for g in zip(*(grads[k] for k in ('risk', 'decision', 'joint')))]
                relations = {k: method.geometry(select(grads['risk']), select(grads[k[5:]]))
                    for k in ('risk_rank0', 'risk_rank1999', 'risk_point0', 'risk_point1999')}
                relations['base_rank1999'] = method.geometry(base, grads['rank1999'])
                steps = {}
                if arm == 'paired_rank':
                    for variant in method.VARIANTS:
                        changed, detail = method.virtual(state, grads, names, variant)
                        steps[variant] = dict(detail,
                            fitting=api.full(api.model_from(changed), data, state['sampler']['counts'], guard),
                            joint_attribution=api.attribution(changed, data, guard))
                source.checked(ref)
                rows.append(dict(job=job['key'], arm=arm, checkpoint=ref,
                    full_fitting_queries=len(data['causal']['groups']), losses=losses, relations=relations,
                    virtual_steps=steps, original_fitting=receipt['final_fitting'],
                    original_joint_attribution=receipt['joint_attribution'], source_sites=state['source_sites']))
                beat(state='gradient_complete', complete=len(rows), total=9)
                del state, grads; source.gc.collect(); guard()
            del data, origin; source.gc.collect()
        rank = [r for r in rows if r['arm'] == 'paired_rank']
        conflicting = sum(all(r['relations'][k]['first_order_conflict'] is True
            for k in ('risk_rank0', 'risk_rank1999')) for r in rank)
        improved = 0
        for r in rank:
            a, b = (r['virtual_steps'][k] for k in ('rank', 'risk_protected_rank'))
            improved += b['fitting']['mean_MSE'] < a['fitting']['mean_MSE'] and (
                b['joint_attribution']['total']['total_joint_cost'] <= a['joint_attribution']['total']['total_joint_cost']+1e-12)
        result = source.stable(dict(status='fresh_run_own_fitting_gradients_and_disposable_updates', rows=rows,
            rank_conflicting_contexts=int(conflicting), protected_better_contexts=int(improved),
            protected_training_followup_supported=conflicting >= 2 and improved >= 2,
            persisted_training_updates=0, disposable_optimizer_updates=12, no_new_V_or_H=True,
            original_rules_unchanged=True, identity_certification=False, independent_confirmation=False,
            deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
        if args.replay:
            api.base.exact(source.read(PUBLIC/'analysis.json'), result)
            api.base.exact(source.read(PUBLIC/'verification.json')['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
            api.base.exact(before, {str(f): source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()})
            beat(state='complete', readonly_replay=True)
            print(source.json.dumps(dict(exact=True, public_files_unchanged=len(before),
                seconds=time.monotonic()-started, pid=source.os.getpid(),
                peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss)))
            return
        source.once(PUBLIC/'analysis.json', result)
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all9_gradients_and12_virtual_updates_exact=True,
                analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        source.once(PUBLIC/('revised_verify_execution.json' if args.verify else 'revised_execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        report(result); beat(state='complete')
        print(source.json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
