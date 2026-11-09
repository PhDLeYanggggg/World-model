"""Build an anonymous scalar replay for the matched pooled-coverage readout."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_cv_pooled_coverage as study

np,base,prior = study.np,study.base,study.prior


def collect(frozen):
    _,donor,_ = base.register()
    data = base.fit_population(lambda **v:None)
    scorer = prior.prior.parent.readout
    rows = []
    contrast = {f'{a}_vs_{k}':(b,a) for a in study.ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    contrast.update(pool_vs_gain=('gain_match','pool_full'),pool_vs_marginal=('marginal','pool_full'),
                    pool_vs_unpooled=('unpooled','pool_full'))
    for c in frozen['cells']:
        ident = c['identity']
        hi,h = base.subset(data,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
        bank = base.expanded(cv,raw)
        with np.load(study.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            actions = {k:z[k] for k in study.ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            c0 = cv[ix]
            paths = {base.CV:c0,base.OLS:ols[ix],**{k:prior.apply(c0,bank[ix],a[ix]) for k,a in actions.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            terms = {k:prior.study.motion.full_masses(paths[a],paths[b],c0,y,v,1./n[inv]) for k,(a,b) in contrast.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=terms))
    return rows


def build(path):
    result,freeze = study.OUT/'readout.json',study.OUT/'freeze.json'
    r,f = study.read(result),study.read(freeze)
    if r['freeze']!=study.sha(freeze) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require matching fitting-only experiment')
    rows = collect(f)
    roster = {v['recording']:v['source'] for v in rows}
    if prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)!=r['summary']:
        raise ValueError('Export input does not reproduce the original public table')
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(schema_version=1,seeds=list(seeds.values()),roster={rec[k]:src[v] for k,v in roster.items()},
        recording_studies={'pooled':pack.trim(rows,rec,src,seeds)},
        joint_cells={k:{str(s):{} for s in seeds.values()} for k in src.values()})
    expected = dict(pooled={**r['summary'],'seeds':list(seeds.values())},joint_pair_contrasts={})
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)')!=1:
        raise ValueError('Unexpected replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)') + '''
root = Path(__file__).resolve().parent
verify_manifest(root)
result = recompute(read(root/'statistics.json'))
assert result == read(root/'expected.json')
print(json.dumps(result['pooled'],sort_keys=True))
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py with NumPy. Replays pooled scalar utility and risk, not geometry or solver.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    print(study.sha(path))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_pooled_coverage.zip')
    build(p.parse_args().out)
