"""Own-fitting source-disjoint causal interaction probes; no new V or H."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_rollout_history as run
from src.world_model import m3w_candidate_neighbor_basis as features
from src.evaluation import m3w_candidate_neighbor_probe as method
from src.evaluation.m3w_decision_semantics import diagnose

source, api = run.source, run.api
np = source.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_candidate_neighbor_v1'
FILES = (Path(__file__), ROOT/'src/world_model/m3w_candidate_neighbor_basis.py',
    ROOT/'src/evaluation/m3w_candidate_neighbor_probe.py', ROOT/'tests/test_m3w_candidate_neighbor_probe.py',
    PUBLIC/'protocol.md')


def frozen_slices(state, data, added, guard):
    c = data['causal']; z = api.parent.kin.batch_features(api.extended(data, state['feature_arm']), 'kinematic')
    model = api.model_from(state)
    r, d = (np.empty((len(c['ids']), 7, 2), np.float32) for _ in range(2))
    for ix, seg in api.parent.batches(c):
        guard()
        with api.torch.no_grad():
            a, b = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
        r[ix], d[ix] = a.numpy(), b.numpy()
    y = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        api.torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    w = api.parent.query_weights(c, state['sampler']['counts'])
    all_metrics = api.previous.previous.attribute(r, d, y, c['allowed'], data['observed_future'], w)
    support = c['allowed'] & data['observed_future'][:, None]
    coordinate = added[..., 20]; proposal = c['allowed']
    # Causal own-B proposals determine cuts; outcome availability cannot move them.
    cuts = np.quantile(coordinate[proposal], [.25, .5, .75])
    bins = np.searchsorted(cuts, coordinate, side='right')
    rows = []
    for index in range(4):
        at = proposal & (bins == index)
        rows.append(dict(quartile=index, proposal_count=int(at.sum()),
            known_count=int((at & support).sum()),
            **diagnose(r, d, y, at, data['observed_future'], w)))
    return dict(cuts=cuts.tolist(), total=all_metrics, slices=rows,
        causal_axis='candidate_minus_floor_minimum_predicted_neighbor_distance',
        cost_weighting='original_fitting_sampler_mass_and_parent_target_RMS',
        descriptive_only=True)


def audit(job, values, guard, verify):
    data, _ = run.inputs(values, job); c = data['causal']
    ref = source.read(run.PUBLIC/'fits'/(job['key']+'_relational.json'))
    state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
    added = features.chunked(c['x']); slices = frozen_slices(state, data, added, guard)
    api.base.exact(slices['total'], ref['joint_attribution'])
    if (c['allowed'].sum(1) > 1).any():
        raise ValueError('Exactly original one-proposal mask required')
    row, action = np.nonzero(c['allowed'] & data['observed_future'][:, None])
    common = np.concatenate((api.parent.kin.motion_basis(c['x']), api.basis(c['x'])), -1)
    target = api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']).numpy()
    query = np.full(len(c['ids']), -1, int)
    for index, ids in enumerate(c['groups']):
        query[ids] = index
    if (query < 0).any():
        raise ValueError('Original complete query assignment required')
    result, arrays = method.probe(common[row, action].astype(float), added[row, action].astype(float),
        target[row, action].astype(float), c['sites'][row], query[row])
    arrays.update(ids=c['ids'][row], actions=action,
        all_neighbor_basis=added, original_proposal=c['allowed'])
    path = PRIVATE/(job['key']+'_probes.npz')
    hashes = {key: api.parent.kin.digest(value) for key, value in arrays.items()}
    if verify:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError('Exact saved probe keys required')
            for key in arrays:
                api.base.exact(saved[key], arrays[key])
    else:
        if path.exists():
            raise ValueError('Existing probe output requires --verify; do not overwrite')
        np.savez_compressed(path, **arrays)
    source.checked(ref['checkpoint'])
    return dict(job=job['key'], result=result, own_B_rows=len(c['ids']),
        known_proposed_rows=len(row), unknown_proposed_rows=int((c['allowed'] & ~data['observed_future'][:, None]).sum()),
        source_names=sorted(set(c['sites'])), ids_hash=api.parent.kin.digest(c['ids']),
        input_hash=state['input_hash'], parent_checkpoint=ref['checkpoint'],
        saved_probes=source.artifact(path), array_hashes=hashes, frozen_slices=slices,
        valid_aligned_neighbor_row_fraction=float((added[:, 0, 0] > 0).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'run', 'verify')); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = run.old.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), args.phase)
        _, values = run.register(beat)
        api.base.exact(source.read(run.PUBLIC/'verification.json')['analysis_sha256'], source.sha(run.PUBLIC/'analysis.json'))
        reg = PUBLIC/'registration_v3.json'
        source.once(reg, dict(parent=source.artifact(run.REG), parent_result=source.artifact(run.PUBLIC/'analysis.json'),
            files={str(f.relative_to(ROOT)): source.sha(f) for f in FILES}, feature_names=list(features.NAMES),
            initial_registration=source.artifact(PUBLIC/'registration.json'),
            intermediate_registration=source.artifact(PUBLIC/'registration_v2.json'),
            correction='JSON-stable feature-name list and exact original attribution helper; no probe result yet',
            jobs=[j['key'] for j in values[3]], ridge_penalty=.01, own_B_only=True,
            followup_screen='MSE improves >=1% and joint sign-cost nonworse in >=2/3 contexts',
            new_V_or_H=False, original_rules_unchanged=True, identity_certification=False))
        if args.phase == 'register':
            return
        for f in (*FILES, reg):
            saved = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(saved).hexdigest() != source.sha(f):
                raise ValueError('Commit prospective protocol and code before probing real data')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase immutable')
        rows = []
        for job in values[3]:
            row = audit(job, values, guard, args.phase == 'verify'); rows.append(row)
            source.once(PUBLIC/(job['key']+'.json'), source.stable(row))
            beat(state='own_B_context_complete', complete=len(rows), total=3)
            source.gc.collect(); guard()
        count = sum(r['result'].get('improves_MSE_and_nonworse_cost', False) for r in rows)
        result = source.stable(dict(status='fresh_run_own_B_leave_source_probe', contexts=rows,
            followup_neural_feature_training_supported=count >= 2, passing_contexts=count,
            new_neural_training=False, new_V_or_H=False, original_rules_unchanged=True,
            identity_certification=False, deployment_changed=False, independent_confirmation=False,
            submission_ready=False, stage5c_executed=False, smc_enabled=False))
        source.once(PUBLIC/'analysis.json', result)
        if args.phase == 'verify':
            source.once(PUBLIC/'verification.json', dict(all3_metrics_and_probe_coefficients_exact=True,
                parent_checkpoints_unchanged=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            no_new_V_or_H=True, neural_training=False))
        beat(state='complete', passing_contexts=count)


if __name__ == '__main__':
    main()
