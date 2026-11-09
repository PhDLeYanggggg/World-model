"""Build a NumPy-only anonymous replay of matched policy and equal-radius tables."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_recipient_radius as radius
from scripts import build_m3w_review_replay as pack

study,exp,np,base,prior = radius.study,radius.exp,radius.np,radius.base,radius.prior


def matched(cells,d,donor):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in study.ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(study.PRIMARY)
    rows = []
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in study.ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            c0 = cv[ix]
            paths = {base.CV:c0,base.OLS:ols[ix],**{k:exp.slate.path_for(c0,b[ix],a[k][ix]) for k in study.ARMS}}
            z = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = exp.weight(q)
            terms = {k:prior.study.motion.full_masses(paths[l],paths[r],c0,y,v,w) for k,(l,r) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(z,source=str(source),seed=seed,comparisons=terms))
    return rows


def build(path):
    f,r,a = [study.OUT/k for k in ('freeze.json','readout.json','radius_control.json')]
    freeze,main,control = map(exp.read,(f,r,a))
    if (main['freeze']!=exp.sha(f) or control['parent']['sha256']!=exp.sha(r)
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct'] for z in (main,control))):
        raise ValueError('Require frozen, development-only readouts')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    rows = dict(matched=matched(freeze['cells'],d,donor),radius=radius.collect(freeze['cells'],d,donor)[0])
    roster = {x['recording']:x['source'] for x in rows['matched']}
    expected = {'matched':main['summary'],'radius':control['summary']}
    for k,v in rows.items():
        if prior.prefix.grouped.grouped_readout(v,roster,base.SEEDS)!=expected[k]:
            raise ValueError('Recomputed policy table differs from verified aggregate')
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(schema_version=1,roster={rec[k]:src[v] for k,v in roster.items()},seeds=list(seeds.values()),
        recording_studies={k:pack.trim(v,rec,src,seeds) for k,v in rows.items()},
        joint_cells={k:{str(i):{} for i in seeds.values()} for k in src.values()})
    expected = {k:dict(v,seeds=list(seeds.values())) for k,v in expected.items()}
    expected['joint_pair_contrasts'] = {}
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)')!=1:
        raise ValueError('Unexpected reviewer replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)')+'''
root = Path(__file__).resolve().parent
verify_manifest(root)
actual = recompute(read(root/'statistics.json'))
if actual!=read(root/'expected.json'):
    raise ValueError('Scalar readouts did not replay')
print('Exact anonymous replay: matched and equal-radius tables, 72 views each.')
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py. Replays scalar risks and utility, not training or path geometry.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    print(exp.sha(path),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_recipient_path.zip')
    build(p.parse_args().out)
