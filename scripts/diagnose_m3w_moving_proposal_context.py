"""Post-fit own-B source/context and bridge-removal diagnostics, no selection."""
import copy
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_proposal_context as run
from scripts.replay_m3w_moving_paired_decision import manifest

api, source = run.api, run.source


def fitting_slices(state, data, guard):
    c = data['causal']; z = api.validate(data, state); model = api.model_from(state)
    r, d = (api.np.empty((len(c['ids']), 7, 2), api.np.float32) for _ in range(2))
    for ix, seg in api.parent.parent.batches(c):
        guard()
        with api.torch.no_grad():
            a, b = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
        r[ix], d[ix] = a.numpy(), b.numpy()
    y = (api.parent.parent.kin.parent.signed(data['truth_by_arm']['bound_aware'])/
        api.torch.from_numpy(api.parent.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    w = api.parent.parent.query_weights(c, state['sampler']['counts'])
    proposal = api.proposal_from_mask(c['allowed']); mixed = api.np.zeros(len(proposal), bool)
    for g in c['groups']:
        mixed[g] = len(set(proposal[g][proposal[g] >= 0])) > 1
    masks = {str(s):c['sites'] == s for s in sorted(set(c['sites']))}
    masks.update(mixed_proposal_query=mixed, single_or_empty_proposal_query=~mixed)
    out = {}
    for name, at in masks.items():
        weight = w*at; mass = float(weight.sum())
        allowed = c['allowed'] & at[:, None]
        out[name] = dict(rows=int(at.sum()), active_weight_mass=mass,
            joint=api.parent.previous.previous.attribute(r, d, y, allowed, data['observed_future'], weight/mass)
                if mass > 0 else None,
            MSE=float((((r-y)**2).mean(-1)*allowed).sum(1)@weight/mass) if mass > 0 else None)
    return out


def remove_bridge(state):
    result = copy.deepcopy(state)
    for head in ('risk', 'decision'):
        result['model'][head+'.bridge.weight'].zero_()
    return result


def main():
    if not source.read(run.PUBLIC/'verification.json')['all6_fitting_metrics_and_attributions_exact']:
        raise ValueError('Completed frozen fit and replay required')
    before = manifest(run.PUBLIC)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        runtime = run.old.old.old.previous.teacher.lineage; runtime.PRIVATE = run.PRIVATE
        started, guard, beat = runtime.runtime({}, source.read(run.CONFIG), 'fitting_diagnosis')
        _, values = run.register(beat); rows = []
        for job in values[3]:
            data, _ = run.inputs(values, job)
            for arm in api.ARMS:
                ref = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                no_bridge = remove_bridge(state)
                rows.append(dict(job=job['key'], arm=arm,
                    bridge_weight_norms={h:float(state['model'][h+'.bridge.weight'].norm()) for h in ('risk', 'decision')},
                    original=ref['joint_attribution'], no_bridge=api.attribution(no_bridge, data, guard),
                    original_fitting=ref['final_fitting'], no_bridge_fitting=api.full(no_bridge, data, guard),
                    slices=fitting_slices(state, data, guard)))
                beat(state='fitting_diagnosis', complete=len(rows))
            del data; source.gc.collect(); guard()
        api.base.exact(before, manifest(run.PUBLIC))
        result = source.stable(dict(status='fresh_run_posthoc_fitting_diagnostic', rows=rows,
            pid=source.os.getpid(), seconds=time.monotonic()-started, prior_public_files_unchanged=len(before),
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            models_retrained=False, causal_contribution_claim=False, used_for_selection=False,
            independent_confirmation=False, additional_held_readout=False,
            interpretation='Removing a fitted branch is descriptive, not a retrained ablation or safety evidence'))
        source.once(run.PUBLIC/'fitting_diagnosis.json', result); beat(state='complete')
        print(source.json.dumps(dict(pid=result['pid'], rows=len(rows), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
