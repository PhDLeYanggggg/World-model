"""Fresh fitting-only audit of proposal masks and cross-channel decisions."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_composed_risk as previous
from src.evaluation import m3w_decision_semantics as diagnostic

source, api = previous.source, previous.api.parent
direct = previous.previous.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_decision_semantics_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_decision_semantics_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        previous.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = previous.old.previous.teacher.lineage.runtime({}, source.read(previous.CONFIG), 'decision_semantics')
        _, values = previous.register(beat)
        _, _, all_data, jobs, refs, parts, _, routers, *_ = values
        source.once(PUBLIC/'registration_v2.json', dict(parent=source.artifact(previous.REG),
            files={str(p.relative_to(ROOT)): source.sha(p) for p in
                (Path(__file__), ROOT/'src/evaluation/m3w_decision_semantics.py', ROOT/'tests/test_m3w_decision_semantics.py')},
            own_B_only=True, no_new_V_or_H=True, no_new_training=True, original_rules_unchanged=True))
        rows = []
        for job in jobs:
            data, warm = previous.previous.previous.inputs(values, job)
            c = data['causal']; part = parts[job['key']][0]
            raw = previous.old.packet(all_data, job, refs, part, training=False)
            state = api.base.read_checkpoint(source.checked(routers[job['key'], 1]['checkpoint']))
            direct.api.parent.assert_excluded(state, raw['causal'], part['names'])
            logits = direct.api.parent.router.predict(state, raw['causal'])
            proposal, _ = direct.api.parent.router.choose(logits, raw['causal']['allowed'])
            mask = diagnostic.proposal_mask(proposal, raw['causal']['allowed'])
            source.np.testing.assert_array_equal(c['ids'], raw['ids'])
            source.np.testing.assert_array_equal(c['x'], raw['causal']['x'])
            source.np.testing.assert_array_equal(c['allowed'], mask)
            support = dict(rows=len(mask), raw_legal_actions=int(raw['causal']['allowed'].sum()),
                supervised_actions=int(mask.sum()), max_supervised_actions_per_row=int(mask.sum(1).max()),
                router_fallback_rows=int((proposal < 0).sum()),
                exact_frozen_router_mask=True, mask_hash=source.array_hash(mask),
                router=routers[job['key'], 1]['checkpoint'])
            z = api.kin.batch_features(data, 'kinematic')
            # Predictions and causal proposal support are fixed before target statistics.
            predictions = {}
            for arm in previous.api.ARMS:
                receipt = source.read(previous.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                fitted = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
                api.validate(data, fitted)
                weights = api.query_weights(c, fitted['sampler']['counts'])
                model = api.model_from(fitted)
                risk = source.np.empty((len(c['ids']), 7, 2), source.np.float32)
                decision = source.np.empty_like(risk)
                for ix, seg in api.batches(c):
                    guard()
                    with api.torch.no_grad():
                        r, d = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
                    risk[ix], decision[ix] = r.numpy(), d.numpy()
                predictions[arm] = risk, decision
            y = (api.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
                api.torch.from_numpy(api.kin.parent.normalizer(data['preprocess']))).numpy()
            metrics = {arm: diagnostic.diagnose(r, d, y, mask, data['observed_future'], weights)
                for arm, (r, d) in predictions.items()}
            by_source = {str(site): {arm: diagnostic.diagnose(r, d, y,
                mask & (c['sites'] == site)[:, None], data['observed_future'], weights)
                for arm, (r, d) in predictions.items()} for site in sorted(set(c['sites']))}
            rows.append(dict(job=job['key'], support=support, metrics=metrics, by_source=by_source))
            del raw, data, z, state, fitted, model, predictions, warm
            source.gc.collect(); beat(state='context_complete', job=job['key'])
        output = PUBLIC/'analysis.json'
        if args.verify and not output.exists():
            raise ValueError('Original diagnosis required')
        source.once(output, source.stable(dict(status='fresh_run_fitting_only', rows=rows,
            all_action_supervision_hypothesis_refuted=all(r['support']['exact_frozen_router_mask'] for r in rows),
            no_new_V_or_H=True, no_new_training=True, original_rules_unchanged=True,
            identity_certification=False, deployment_changed=False, submission_ready=False)))
        if args.verify:
            source.once(PUBLIC/'verification.json', dict(all3_exact=True, analysis_sha256=source.sha(output)))
        source.once(PUBLIC/('verify_execution.json' if args.verify else 'diagnose_execution.json'), dict(
            pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete')


if __name__ == '__main__':
    main()
