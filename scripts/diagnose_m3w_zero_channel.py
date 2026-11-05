"""Attribute existing own-B rejections without reading new held outcomes."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_decision_semantics as prior
from src.evaluation.m3w_zero_channel import attribute

previous, source, api = prior.previous, prior.source, prior.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_zero_channel_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_zero_channel_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        previous.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = previous.old.previous.teacher.lineage.runtime({}, source.read(previous.CONFIG), 'zero_channel')
        _, values = previous.register(beat)
        api.base.exact(source.read(prior.PUBLIC/'verification.json')['analysis_sha256'], source.sha(prior.PUBLIC/'analysis.json'))
        source.once(PUBLIC/'registration.json', dict(parent=source.artifact(prior.PUBLIC/'analysis.json'),
            files={str(f.relative_to(ROOT)): source.sha(f) for f in (Path(__file__),
                ROOT/'src/evaluation/m3w_zero_channel.py', ROOT/'tests/test_m3w_zero_channel.py')},
            no_new_training=True, no_new_V_or_H=True, own_B_only=True, original_rules_unchanged=True))
        rows = []
        for job in values[3]:
            data, warm = previous.previous.previous.inputs(values, job)
            c = data['causal']; z = api.kin.batch_features(data, 'kinematic')
            predictions = {}
            for arm in previous.api.ARMS:
                receipt = source.read(previous.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
                api.validate(data, state); weights = api.query_weights(c, state['sampler']['counts'])
                model = api.model_from(state)
                r, d = (source.np.empty((len(c['ids']), 7, 2), source.np.float32) for _ in range(2))
                for ix, seg in api.batches(c):
                    guard()
                    with api.torch.no_grad():
                        a, b = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
                    r[ix], d[ix] = a.numpy(), b.numpy()
                predictions[arm] = r, d
            y = (api.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
                api.torch.from_numpy(api.kin.parent.normalizer(data['preprocess']))).numpy()
            rows.append(dict(job=job['key'], metrics={arm: attribute(r, d, y, c['allowed'],
                data['observed_future'], weights) for arm, (r, d) in predictions.items()}))
            del data, warm, z, state, model, predictions
            source.gc.collect(); beat(state='context_complete', job=job['key'])
        output = PUBLIC/'analysis.json'
        if args.verify and not output.exists():
            raise ValueError('Original attribution required')
        source.once(output, source.stable(dict(status='fresh_run_fitting_only', rows=rows,
            no_new_training=True, no_new_V_or_H=True, original_rules_unchanged=True)))
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all6_exact=True, analysis_sha256=source.sha(output)))
        source.once(PUBLIC/('verify_execution.json' if args.verify else 'diagnose_execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__':
    main()
