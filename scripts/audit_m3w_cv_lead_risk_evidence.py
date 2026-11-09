"""Export anonymous matched risk/utility totals and localize observed failures."""
import hashlib
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_cv_lead_risk as study
from scripts.audit_m3w_cv_signed_failures import flags

np,base,prior = study.np,study.base,study.prior
read,sha,once,stable,artifact,checked = study.read,study.sha,study.once,study.stable,study.artifact,study.checked


def collect(f):
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    scorer = prior.prior.parent.readout
    contrasts = {f'{a}_vs_{tag}':(b,a) for a in study.ARMS for tag,b in (('cv',base.CV),('ols4',base.OLS))}
    contrasts.update(lead_matched=('cond_flat_matched','cond_lead_matched'),
        tail_matched=('cond_lead_matched','tail_lead_matched'),lead_under_tail=('tail_flat_matched','tail_lead_matched'))
    counts = {k:Counter() for k in (*study.MATCHED,'legacy','cond_lead','tail_lead')}
    rows = []
    for c in f['cells']:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            actions = {k:z[k] for k in study.ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            c0 = cv[ix]
            paths = {base.CV:c0,base.OLS:ols[ix],**{k:prior.apply(c0,bank[ix],a[ix]) for k,a in actions.items()}}
            z = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            for k,stats in counts.items():
                stats.update(flags(z['metrics'][k]['risk']))
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            terms = {k:prior.study.motion.full_masses(paths[a],paths[b],c0,y,v,1./n[inv]) for k,(a,b) in contrasts.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(z,source=str(source),seed=seed,comparisons=terms))
    return rows,counts


def package(rows,roster,summary,out):
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(schema_version=1,seeds=list(seeds.values()),roster={rec[k]:src[v] for k,v in roster.items()},
        recording_studies={'lead_risk':pack.trim(rows,rec,src,seeds)},
        joint_cells={k:{str(s):{} for s in seeds.values()} for k in src.values()})
    expected = dict(lead_risk={**summary,'seeds':list(seeds.values())},joint_pair_contrasts={})
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)')!=1:
        raise ValueError('Unexpected replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)') + '''
root = Path(__file__).resolve().parent
verify_manifest(root)
result = recompute(read(root/'statistics.json'))
assert result == read(root/'expected.json')
print(json.dumps(result['lead_risk'],sort_keys=True))
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py with NumPy. This replays scalar risk and utility only.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(out,files)


def main():
    out = study.OUT
    fp,rp = out/'freeze.json',out/'readout.json'
    f,r = read(fp),read(rp)
    if (r['freeze']!=sha(fp) or not f['before_held_scoring'] or r['a_labels_opened']
            or r['independent_confirmation'] or not r['original_2pct']):
        raise ValueError('Expected fitting-only frozen scores')
    rows,counts = collect(f)
    roster = {v['recording']:v['source'] for v in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if summary!=r['summary']:
        raise ValueError('Anonymous scalar inputs do not reproduce the primary table')
    path = study.PRIVATE/'anonymous_lead_risk.zip'
    package(rows,roster,summary,path)
    once(out/'failure_replay.json',stable(dict(parent=artifact(rp),code=artifact(Path(__file__)),
        risk_failures={k:dict(v) for k,v in counts.items()},anonymous_sha256=sha(path),
        known_only_diagnostic_not_a_replacement=True,original_2pct=True,independent_confirmation=False)))
    print(sha(path))


if __name__=='__main__':
    main()
