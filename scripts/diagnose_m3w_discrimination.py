"""Separate own-B ranking ability from fixed-zero decision errors."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_harm_floor as parent
from src.evaluation.m3w_discrimination_audit import audit

source, api = parent.source, parent.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_discrimination_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_discrimination_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        parent.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = parent.old.previous.teacher.lineage.runtime({}, source.read(parent.CONFIG), 'discrimination')
        _, values = parent.register(beat)
        source.once(PUBLIC/'registration.json', dict(
            files={str(p.relative_to(ROOT)): source.sha(p) for p in (Path(__file__),
                ROOT/'src/evaluation/m3w_discrimination_audit.py', ROOT/'tests/test_m3w_discrimination_audit.py')},
            parent=source.artifact(parent.PUBLIC/'verification.json'), own_B_only=True,
            no_threshold_search=True, no_model_training=True, no_new_V_or_H=True, exploratory=True))
        rows = []
        for job in values[3]:
            data, origin = parent.inputs(values, job); c = data['causal']
            for arm in api.ARMS:
                ref = source.read(parent.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                z = api.parent.validate(data, state); model = api.model_from(state)
                weights = api.parent.query_weights(c, state['sampler']['counts'])
                risk, decision = (source.np.empty((len(c['ids']), 7, 2), source.np.float32) for _ in range(2))
                for ix, seg in api.parent.batches(c):
                    guard()
                    with api.torch.no_grad():
                        r, d = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
                    risk[ix], decision[ix] = r.numpy(), d.numpy()
                y = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
                    api.torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).numpy()
                result = audit(risk, decision, y, c['allowed'], data['observed_future'], weights, c['sites'])
                api.base.exact(result['harmful_admitted_both'], ref['joint_attribution']['total']['false_safe_count'])
                rows.append(dict(job=job['key'], arm=arm, **result))
                beat(state='discrimination_complete', complete=len(rows), total=9)
            del data, origin, state, model, z, risk, decision, y; source.gc.collect(); guard()
        path = PUBLIC/'analysis.json'
        if args.verify and not path.exists():
            raise ValueError('Initial diagnosis required before verification')
        source.once(path, source.stable(dict(status='fresh_run_own_fitting_diagnostic', rows=rows,
            no_new_V_or_H=True, original_rules_unchanged=True, no_identity_certification=True)))
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all9_exact=True, analysis_sha256=source.sha(path)))
        source.once(PUBLIC/('verify_execution.json' if args.verify else 'execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__':
    main()
