"""Post-fit own-B admission tradeoff; no new fitting or held readout."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_harm_floor as run
from src.evaluation.m3w_harm_admission import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    source, api, public = run.source, run.api, run.PUBLIC
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'admission_tradeoff')
        _, values = run.register(beat)
        api.base.exact(source.read(public/'verification.json')['analysis_sha256'], source.sha(public/'analysis.json'))
        source.once(public/'admission_diagnosis_registration.json', dict(
            parent=source.artifact(public/'analysis.json'), posthoc_descriptive=True,
            files={str(f.relative_to(ROOT)): source.sha(f) for f in (Path(__file__),
                ROOT/'src/evaluation/m3w_harm_admission.py', ROOT/'tests/test_m3w_harm_admission.py')},
            own_B_only=True, no_new_training=True, no_new_V_or_H=True, original_rules_unchanged=True))
        rows = []
        for job in values[3]:
            data, warm = run.inputs(values, job); c = data['causal']; scores, receipts = {}, {}
            for arm in api.ARMS:
                receipt = source.read(public/'fits'/(job['key']+'_'+arm+'.json')); receipts[arm] = receipt
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
            for before, after in (('joint', 'harm_floor'), ('uniform_harm', 'harm_floor')):
                base = scores[before], scores[after], y
                total = analyze(*base, c['allowed'], data['observed_future'], weights)
                a, b = (receipts[arm]['joint_attribution']['total'] for arm in (before, after))
                api.base.exact(total['before_false_safe'], a['false_safe_count'])
                api.base.exact(total['after_false_safe'], b['false_safe_count'])
                source.np.testing.assert_allclose(total['cost_change'], b['total_joint_cost']-a['total_joint_cost'],
                    rtol=1e-10, atol=1e-12)
                by_source = {str(site): analyze(*base, c['allowed'] & (c['sites'] == site)[:, None],
                    data['observed_future'], weights) for site in sorted(set(c['sites']))}
                rows.append(dict(job=job['key'], comparison=after+'_minus_'+before, total=total, by_source=by_source))
            beat(state='tradeoff_complete', completed=len(rows))
            del data, warm, state, model, z, scores, out, y, base
            source.gc.collect(); guard()
        outpath = public/'admission_diagnosis.json'
        if args.verify and not outpath.exists():
            raise ValueError('Initial descriptive diagnosis required')
        source.once(outpath, source.stable(dict(status='fresh_run_fitting_only_posthoc', rows=rows,
            no_new_training=True, no_new_V_or_H=True, original_rules_unchanged=True)))
        if args.verify:
            source.once(public/'admission_diagnosis_verification.json', dict(all6_exact=True,
                analysis_sha256=source.sha(outpath)))
        suffix = 'verify' if args.verify else 'initial'
        source.once(public/('admission_diagnosis_'+suffix+'_execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__':
    main()
