"""Whole-fitting task gradients and disposable saved-AdamW counterfactuals."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_scale_ablation as run
from src.evaluation import m3w_all_easy_gradient as method

source = run.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_all_easy_gradient_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_all_easy_gradient_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = PRIVATE
        start, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'diagnose')
        _, values = run.register(beat)
        run.api.base.exact(source.read(run.PUBLIC/'verification.json')['analysis_sha256'], source.sha(run.PUBLIC/'analysis.json'))
        source.once(PUBLIC/'registration.json', dict(
            files={str(p.relative_to(ROOT)): source.sha(p) for p in (Path(__file__),
                ROOT/'src/evaluation/m3w_all_easy_gradient.py', ROOT/'tests/test_m3w_all_easy_gradient.py', PUBLIC/'protocol.md')},
            previous_result=source.artifact(run.PUBLIC/'analysis.json'),
            main_models='three_unmasked_MSE_contexts', diagnostic_only=True,
            followup_rule='at_least2of3_joint_first_order_easy_harm_and2of3_actual_protected_step_better',
            original_rules_unchanged=True, no_new_V_or_H_outcome_readout=True))
        if args.verify and not (PUBLIC/'analysis.json').exists():
            raise ValueError('An original diagnostic is required for replay')
        results = []
        for job in values[3]:
            d, warm = run.parent.prior.student_data(values, job)
            references = [(name, directory) for name, directory in (('unmasked_MSE', run.parent.PUBLIC), ('scale_ablated_MSE', run.PUBLIC))]
            warm_ref = values[-1][job['key'], 'kinematic']['checkpoint']
            source.checked(warm_ref)
            models = [('warm', warm, d, warm_ref)]
            for name, directory in references:
                ref = source.read(directory/'fits'/(job['key']+'_continued_mse.json'))['checkpoint']
                models.append((name, run.api.base.read_checkpoint(source.checked(ref)),
                    run.api.prepare(d) if name.startswith('scale_') else d, ref))
            for name, state, data, ref in models:
                model = method.kin.KinematicRiskHead(state['settings']['width'], state['seed'])
                model.load_state_dict(state['model']); model.eval()
                before, gradients = method.full(model, data, warm['sampler']['counts'], True, guard)
                steps = {}
                for variant in method.VARIANTS:
                    changed, detail = method.virtual(model, state, gradients, variant)
                    after, _ = method.full(changed, data, warm['sampler']['counts'], False, guard)
                    steps[variant] = dict(detail, after=after)
                    del changed
                run.api.base.exact(model.state_dict(), state['model'])
                if ref:
                    source.checked(ref)
                results.append(dict(job=job['key'], model=name, checkpoint=ref, before=before,
                    virtual_steps=steps, source_sites=state['source_sites'], full_fitting_queries=len(d['causal']['groups']),
                    query_counts_hash=method.kin.digest(warm['sampler']['counts']),
                    input_hash=method.kin.digest(method.kin.batch_features(data, 'kinematic').numpy()),
                    target_hash=method.kin.digest(d['truth_by_arm']['bound_aware'].numpy())))
                beat(state='full_fitting_task_diagnostic', completed=len(results), total=9)
                del model, gradients
            del d, warm, models; source.gc.collect()
        mains = [r for r in results if r['model'] == 'unmasked_MSE']
        first_order = sum(r['virtual_steps']['joint']['shared_relation']['joint_first_order_easy_harm'] is True for r in mains)
        protected_better = sum(r['virtual_steps']['easy_protected']['after']['task_MSE'][1] <
            r['virtual_steps']['joint']['after']['task_MSE'][1] for r in mains)
        result = dict(result_source='fresh_run_fitting_only_gradients_and_virtual_updates', rows=results,
            previous_result=source.artifact(run.PUBLIC/'analysis.json'),
            original_controls_first_order_joint_easy_harm=first_order,
            original_controls_protected_virtual_easy_loss_better=protected_better,
            projected_gradient_followup_supported=first_order >= 2 and protected_better >= 2,
            diagnostic_rule='at_least2of3_original_MSE_contexts_show_joint_easy_harm_and_actual_protection_benefit',
            persisted_training_updates=0, disposable_optimizer_updates=27,
            no_new_V_or_H_outcome_readout=True, original_rules_unchanged=True,
            independent_confirmation=False, deployment_changed=False, submission_ready=False,
            stage5c_executed=False, smc_enabled=False)
        source.once(PUBLIC/'analysis.json', source.stable(result))
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all9_gradients_and27_virtual_updates_exact=True,
                analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        else:
            source.once(PUBLIC/'execution.json', dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
                peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss, completed_models=9))
        lines = ['# Whole-Fitting All/Easy Task Attribution', '',
            'Fresh diagnostic, frozen neural models. Full empirical query-draw weights,',
            'known labels plus unchanged unknown-outcome bounds; only fitting B.',
            'Nine dependent checkpoint views and 27 disposable saved-AdamW steps.',
            'No persisted weights, changed policy, new V/H or independent test.', '',
            '| Context | Model | Shared all/easy cosine | Shared joint/easy cosine | Easy MSE before | Joint step | Protected step | Easy-only step |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in results:
            v = r['virtual_steps']; g = v['joint']['shared_relation']
            lines.append(f"| {r['job']} | {r['model']} | {g['all_easy_cosine']} | {g['joint_easy_cosine']} | {r['before']['task_MSE'][1]:.9g} | {v['joint']['after']['task_MSE'][1]:.9g} | {v['easy_protected']['after']['task_MSE'][1]:.9g} | {v['easy_only']['after']['task_MSE'][1]:.9g} |")
        lines += ['', f"Original MSE joint first-order easy-harm contexts: {first_order}/3.",
            f"Actual projected-step easy MSE better: {protected_better}/3.",
            f"Projected-gradient followup supported: {result['projected_gradient_followup_supported']}.", '',
            'The counterfactual retains AdamW momentum and clipping, so raw gradient',
            'cosine alone is not the actual optimizer effect. A local fitting step',
            'does not prove the cause of the complete training path or held transfer.',
            'The old European membership-BCE/total-cost audit studied a different',
            'task pair and did not support a broad conflict claim; its failure remains.',
            'Original 2% rules and support unchanged. Pixel obs8/pred12, not metric,',
            'seconds, true 3D, foundation or human gold. Stage5C/SMC off;',
            'submission_ready remains false.', '']
        (PUBLIC/'README.md').write_text('\n'.join(lines))
        beat(state='complete')
        print(source.json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
