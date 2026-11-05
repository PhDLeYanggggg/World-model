"""Three frozen own-B controls, eight fixed neighbor permutations, no new V/H."""
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
from src.evaluation import m3w_neighbor_order as method

source, api = run.source, run.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_neighbor_order_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_neighbor_order_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'neighbor_order')
        _, values = run.register(beat)
        files = (Path(__file__), ROOT/'src/evaluation/m3w_neighbor_order.py', PUBLIC/'protocol.md')
        source.once(PUBLIC/'registration_corrected.json', dict(files={str(f.relative_to(ROOT)): source.sha(f) for f in files},
            parent=source.artifact(run.PUBLIC/'verification.json'), own_B_only=True, arms=['joint'],
            initial_registration=source.artifact(PUBLIC/'registration.json'),
            correction='Reuse existing decision cost after first-context identity check rejected initial implementation',
            shifts=list(range(8)), original_rules_unchanged=True, identity_certification=False))
        rows = []
        for job in values[3]:
            data, _ = run.inputs(values, job)
            ref = source.read(run.PUBLIC/'fits'/(job['key']+'_joint.json'))
            state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
            row = method.audit(state, data, guard)
            source.np.testing.assert_allclose(row['shifts'][0]['mean_MSE'], ref['final_fitting']['mean_MSE'], rtol=1e-6, atol=1e-8)
            source.np.testing.assert_allclose(row['shifts'][0]['joint_cost'],
                ref['joint_attribution']['total']['total_joint_cost'], rtol=1e-6, atol=1e-8)
            source.checked(ref['checkpoint'])
            rows.append(dict(row, checkpoint=ref['checkpoint']))
            beat(state='context_complete', complete=len(rows), total=3)
            del data, state; source.gc.collect(); guard()
        sensitive = sum(r['mean_change_over_nonidentity'] is not None and r['mean_change_over_nonidentity'] > .05 for r in rows)
        improved = sum(r['averaging_better_MSE_nonworse_cost'] for r in rows)
        result = source.stable(dict(status='fresh_run_frozen_model_own_B_stress_test', rows=rows,
            sensitive_contexts=sensitive, averaging_better_contexts=improved,
            permutation_training_supported=sensitive >= 2 and improved >= 2,
            training_updates=0, original_rules_unchanged=True, no_new_V_or_H=True, identity_certification=False,
            independent_confirmation=False, deployment_changed=False, submission_ready=False,
            stage5c_executed=False, smc_enabled=False))
        source.once(PUBLIC/'analysis.json', result)
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all3_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        source.once(PUBLIC/('verify_execution.json' if args.verify else 'execution.json'), dict(pid=source.os.getpid(),
            seconds=time.monotonic()-started, peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        lines = ['# Neighbor-Slot Sensitivity', '', 'Complete own-B data; fixed original joint controls; no new V/H.', '',
            '| Context | Changed admissions % | Original / averaged MSE | Original / averaged joint cost |',
            '|---|---:|---|---|']
        for r in rows:
            a, b = r['shifts'][0], r['cyclic_mean']
            lines.append(f"| {r['context']} | {100*r['mean_change_over_nonidentity']:.3f} | {a['mean_MSE']:.7g} / {b['mean_MSE']:.7g} | {a['joint_cost']:.7g} / {b['joint_cost']:.7g} |")
        lines += ['', f'Sensitive contexts: {sensitive}/3. Mean prediction improves both fitting criteria: {improved}/3.',
            f'Prospective permutation-aware training screen: **{result["permutation_training_supported"]}**.', '',
            'Sensitivity is not proof of a bug: the original nearest-neighbor order is a valid convention.',
            'All shifts, clipping and error counts are retained in JSON. Original 2% safety/support',
            'requirements are unchanged. No identity certification or new deployment result.',
            'Native obs8/pred12 image coordinates; Stage5C/SMC off; independent confirmation not_run.', '',
            '[Protocol](protocol.md) | [Full metrics](analysis.json)', '']
        (PUBLIC/'README.md').write_text('\n'.join(lines))
        beat(state='complete'); print(source.json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
