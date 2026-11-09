"""Locate the causal selection gap without changing any camera policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_full_path_selection as run

study, base, np, torch = run.study, run.base, run.np, run.torch
pilot = run.camera.pilot


def predicted(value, legal, q):
    gain = value[..., :2].min(2) > 0.
    risk = value[..., 2:4]-.02*value[..., 4:6]
    z = dict(legal_actions=int(legal.sum()), positive_single_gain=int((legal & gain).sum()),
             nonpositive_all_excess=int((legal & (risk[...,0] <= 0.)).sum()),
             nonpositive_easy_excess=int((legal & (risk[...,1] <= 0.)).sum()),
             positive_and_both_risks=int((legal & gain & (risk <= 0.).all(2)).sum()))
    admissible_pairs = 0
    for j in np.unique(q):
        at = np.flatnonzero(q == j)
        if len(at) < 3:
            continue
        rows, arms = np.nonzero(legal[at])
        a, b = np.triu_indices(len(rows),1)
        i, k = at[rows], arms
        g = value[i,k,:2]
        r = risk[i,k]
        admissible_pairs += int(((rows[a] != rows[b]) & ((g[a]+g[b]) > 0.).all(1) &
                                ((r[a]+r[b]) <= 0.).all(1)).sum())
    return dict(**z, positive_two_gain_and_both_risks=admissible_pairs,
                pairs_ignore_proxy_penalty=True)


def analyze():
    key, _, c, _ = run.register()
    fp, rp = (run.OUT/f'{k}.json' for k in ('freeze','readout'))
    f, r = study.read(fp),study.read(rp)
    if (f['registration'] != key or r['registration'] != key or r['freeze']['sha256'] != base.sha(fp)
            or not f['camera_targets_not_read'] or not f['no_camera_selection']):
        raise ValueError('Require completed frozen camera selection')
    h = c['history']
    q = np.unique(c['q'],return_inverse=True)[1]
    y, v, m = run.camera.targets('wepdtof',c)
    cut = study.read(study.OUT/'pilot_freeze.json')['cut_px']
    out = {}
    for s in study.SEEDS:
        with np.load(study.checked(f['camera_predictions'][str(s)]),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'],c['q'])
            np.testing.assert_array_equal(z['ids'],c['ids'])
            p = {k:z[k] for k in z.files}
        ck = f['value_heads'][str(s)]
        state = torch.load(study.checked(ck['checkpoint']),map_location='cpu',weights_only=True)
        if state['identity'] != ck['identity'] or state['step'] != run.STEPS:
            raise ValueError('Selector does not match frozen path receipt')
        net = study.costs.BridgeValue()
        net.load_state_dict(state['net'])
        net.eval()
        cv,ols = p['cv'],p['ols4']
        raw = np.stack([p[k] for k in ('ols4','median4','clean','noise')],axis=1)
        site = {}
        for label,b in zip(('raw','equal'),run.policy.banks(cv,raw,q),strict=True):
            x,scale,legal,_ = study.costs.features(h,cv,ols,b)
            with torch.inference_mode():
                value = net(torch.from_numpy(x)).numpy()*scale[:,None,None]
            for menu,n in (('analytic',2),('expanded',4)):
                arm = f'{menu}_{label}'
                if label == 'equal':
                    counts,radii = pilot.counts_of(p[arm+'_joint'],cv,q,int(q.max())+1)
                else:
                    counts = np.minimum(2,np.bincount(q)-1)
                    radii = np.zeros(len(counts))
                    np.maximum.at(radii,q,np.linalg.norm(b[:,:n]-cv[:,None],axis=-1).max((1,2)))
                    radii *= counts
                site[arm] = dict(allowed_edits=int(counts.sum()), radius_sum_px=float(radii.sum()),
                    predicted=predicted(value[:,:n],legal[:,:n],q),
                    capacity=pilot.finite(cv,ols,b[:,:n],y,v,q,cut,counts,radii))
                if arm == 'analytic_equal':
                    four = np.broadcast_to(cv[:,None],b.shape).copy()
                    four[:,[0,2]] = b[:,:2]
                    allowed = np.zeros_like(legal)
                    allowed[:,[0,2]] = legal[:,:2]
                    rank = pilot.camera.past_rank(h,cv,four,allowed,q,counts)
                    joint = p[arm+'_joint']
                    run.policy.matched(cv,q,dict(prefix=rank,joint=joint))
                    k = study.metrics(dict(cv=cv,ols4=ols,prefix=rank),y,v,q,base.PILOT,cut,
                                      paired_controls=False)
                    w = 1./np.bincount(q)[q]
                    observed, lower = study.paired(rank,joint,y,v)
                    site[arm]['prefix_control'] = dict(
                        score=k['arms']['prefix'], joint_minus_prefix_known_lower_pp=(
                            100.*np.array([w@observed.sum(1),w@lower.sum(1)])/k['reference_mass']).tolist())
        out[str(s)] = site
        print(dict(seed=s,cells={k:(z['capacity']['solver']['status'],z['predicted']['positive_and_both_risks'])
                                 for k,z in site.items()}),flush=True)
    base.immutable(run.OUT/'failure_analysis.json',dict(registration=key, readout=study.artifact(rp),
        code=study.artifact(Path(__file__)), seeds=out, original_2pct=True,
        labels_hindsight_only_for_capacity=True, independent_confirmation=False, camera_tuning=False))


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    analyze()
