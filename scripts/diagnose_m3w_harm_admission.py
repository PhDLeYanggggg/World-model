"""Own-B attribution of new harmful admissions, no held outcomes or new fitting."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_decision as previous
from src.evaluation.m3w_harm_admission import analyze

source, api = previous.source, previous.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_harm_admission_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_harm_admission_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        previous.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = previous.old.previous.teacher.lineage.runtime({}, source.read(previous.CONFIG), 'harm_admission')
        _, values = previous.register(beat)
        api.base.exact(source.read(previous.PUBLIC/'verification.json')['analysis_sha256'], source.sha(previous.PUBLIC/'analysis.json'))
        source.once(PUBLIC/'registration.json', dict(parent=source.artifact(previous.PUBLIC/'analysis.json'),
            files={str(f.relative_to(ROOT)): source.sha(f) for f in (Path(__file__),
                ROOT/'src/evaluation/m3w_harm_admission.py', ROOT/'tests/test_m3w_harm_admission.py')},
            own_B_only=True, no_new_training=True, no_new_V_or_H=True, original_rules_unchanged=True))
        rows = []
        for job in values[3]:
            data, warm = previous.inputs(values, job); c = data['causal']; scores = {}
            for arm in api.ARMS:
                receipt = source.read(previous.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
                z = api.parent.validate(data, state); model = api.model_from(state)
                weights = api.parent.query_weights(c, state['sampler']['counts'])
                out = source.np.empty((len(c['ids']), 7), source.np.float32)
                for ix, seg in api.parent.batches(c):
                    guard()
                    with api.torch.no_grad():
                        r, d = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
                    out[ix] = api.torch.maximum(r, d).amax(-1).numpy()
                scores[arm] = out
            y = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
                api.torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).numpy()
            base = scores['per_channel'], scores['joint'], y
            total = analyze(*base, c['allowed'], data['observed_future'], weights)
            by_source = {str(site): analyze(*base, c['allowed'] & (c['sites'] == site)[:, None],
                data['observed_future'], weights) for site in sorted(set(c['sites']))}
            rows.append(dict(job=job['key'], total=total, by_source=by_source))
            beat(state='context_complete', job=job['key'], false_safe_before=total['before_false_safe'],
                false_safe_after=total['after_false_safe'], mass_ratio=total['negative_to_positive_mass_ratio'])
            del data, warm, state, model, z, scores, out, y, base
            source.gc.collect(); guard()
        out = PUBLIC/'analysis.json'
        if args.verify and not out.exists():
            raise ValueError('Initial diagnosis required')
        source.once(out, source.stable(dict(status='fresh_run_fitting_only', rows=rows,
            no_new_training=True, no_new_V_or_H=True, original_rules_unchanged=True)))
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all3_exact=True, analysis_sha256=source.sha(out)))
        source.once(PUBLIC/('verify_execution.json' if args.verify else 'diagnose_execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__':
    main()
