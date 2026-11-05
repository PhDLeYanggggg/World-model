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


def context_separation(data, warm, guard):
    c = data['causal']; z = api.validate(data, warm)
    model = api.initialize(warm, 'same_action').eval()
    sums = {arm:dict(squared_difference=0., squared_reference=0., cosine_sum=0., cosine_defined=0, proposals=0)
        for arm in ('same_active', 'proposed_action', 'proposed_all')}
    for ix, seg in api.parent.parent.batches(c):
        guard(); x = z[ix]; segments = api.torch.from_numpy(seg); count = int(seg.max())+1
        with api.torch.no_grad():
            head = model.risk
            code = api.torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
            h = api.F.silu(head.encoder(api.torch.cat((x[..., :428], code), -1))+
                head.kinematics(x[..., 428:492])+head.relation(x[..., 492:524]))
            mask = x[..., -1]
            same, proposed = api.previous.peer_context(h, mask, segments, count)
            active, complete = api.added_contexts(h, mask, segments, count)
            at = mask.bool(); reference = same[at].double()
            for arm, values in (('same_active', active), ('proposed_action', proposed), ('proposed_all', complete)):
                value = values[at].double(); vnorm = value.norm(dim=-1); rnorm = reference.norm(dim=-1)
                defined = (vnorm > 0) & (rnorm > 0); target = sums[arm]
                target['squared_difference'] += float((value-reference).square().sum())
                target['squared_reference'] += float(reference.square().sum())
                target['cosine_sum'] += float(((value*reference).sum(-1)[defined]/(vnorm*rnorm)[defined]).sum())
                target['cosine_defined'] += int(defined.sum()); target['proposals'] += len(value)
    return {arm:dict(**value,
        relative_RMS_difference=float((value['squared_difference']/value['squared_reference'])**.5)
            if value['squared_reference'] > 0 else None,
        cosine_mean=value['cosine_sum']/value['cosine_defined'] if value['cosine_defined'] else None)
        for arm, value in sums.items()}


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
        _, values = run.register(beat); rows = []; separations = []
        for job in values[3]:
            data, warm = run.inputs(values, job)
            separations.append(dict(job=job['key'], causal_parent_context=context_separation(data, warm, guard)))
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
            causal_parent_context_separation=separations,
            pid=source.os.getpid(), seconds=time.monotonic()-started, prior_public_files_unchanged=len(before),
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            models_retrained=False, used_for_selection=False, independent_confirmation=False,
            additional_held_readout=False, interpretation='Fixed fitted cells and source slices are not population causal or safety guarantees'))
        source.once(run.PUBLIC/'fitting_diagnosis.json', result); beat(state='complete')
        print(source.json.dumps(dict(pid=result['pid'], rows=len(rows), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
