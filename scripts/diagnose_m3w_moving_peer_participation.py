"""Own-fitting factorial source/context costs; descriptive, never model selection."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_participation as run
from scripts.replay_m3w_moving_paired_decision import manifest

api, source = run.api, run.source


def fitting_slices(state, data, guard):
    c = data['causal']; r, d = api.normalized_predictions(state, data, guard)
    y = (api.parent.parent.kin.parent.signed(data['truth_by_arm']['bound_aware'])/
        api.torch.from_numpy(api.parent.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    w = api.parent.parent.query_weights(c, state['sampler']['counts'])
    proposal = api.proposal_from_mask(c['allowed']); mixed = api.np.zeros(len(proposal), bool)
    fallback = api.np.zeros(len(proposal), bool)
    for g in c['groups']:
        mixed[g] = len(set(proposal[g][proposal[g] >= 0])) > 1
        fallback[g] = bool((proposal[g] < 0).any())
    masks = {str(s):c['sites'] == s for s in sorted(set(c['sites']))}
    masks.update(mixed_proposal_query=mixed, single_or_empty_proposal_query=~mixed,
        contains_fallback_peer_query=fallback, all_proposing_query=~fallback)
    out = {}
    for name, at in masks.items():
        weight = w*at; mass = float(weight.sum()); allowed = c['allowed'] & at[:, None]
        out[name] = dict(rows=int(at.sum()), active_weight_mass=mass,
            joint=api.parent.previous.previous.attribute(r, d, y, allowed, data['observed_future'], weight/mass)
                if mass > 0 else None,
            MSE=float((((r-y)**2).mean(-1)*allowed).sum(1)@weight/mass) if mass > 0 else None)
    return out


def main():
    if not source.read(run.PUBLIC/'verification.json')['all12_fitting_metrics_and_attributions_exact']:
        raise ValueError('Completed frozen fit and replay required')
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        before = manifest(run.PUBLIC)
        runtime = run.old.old.old.previous.teacher.lineage; runtime.PRIVATE = run.PRIVATE
        started, guard, beat = runtime.runtime({}, source.read(run.CONFIG), 'fitting_diagnosis')
        _, values = run.register(beat); rows = []
        for job in values[3]:
            data, _ = run.inputs(values, job)
            for arm in api.ARMS:
                ref = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                rows.append(dict(job=job['key'], arm=arm, fit_origin=ref['result_source'],
                    original=ref['joint_attribution'], original_fitting=ref['final_fitting'],
                    bridge_weight_norms={h:float(state['model'][h+'.bridge.weight'].norm()) for h in ('risk', 'decision')},
                    slices=fitting_slices(state, data, guard)))
                beat(state='fitting_diagnosis', complete=len(rows), total=12)
            del data; source.gc.collect(); guard()
        api.base.exact(before, manifest(run.PUBLIC))
        result = source.stable(dict(status='fresh_run_posthoc_fitting_diagnostic', rows=rows,
            pid=source.os.getpid(), seconds=time.monotonic()-started, prior_public_files_unchanged=len(before),
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            models_retrained=False, used_for_selection=False, independent_confirmation=False,
            additional_held_readout=False, interpretation='Fixed fitted cells and source slices are not population causal or safety guarantees'))
        source.once(run.PUBLIC/'fitting_diagnosis.json', result); beat(state='complete')
        print(source.json.dumps(dict(pid=result['pid'], rows=len(rows), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
