"""Package matched-exchange recording statistics without trajectories or weights."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import train_m3w_cv_matched_exchange as study

old,base,prior,np,model = study.old,study.base,study.prior,study.np,study.model


def collect(cells,d,donor):
    scorer = prior.prior.parent.readout
    pairs = {f'{m}_vs_{tag}':(base0,m) for m in study.ARMS for tag,base0 in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update({f'{m}_vs_anchor':('half_anchor',m) for m in study.MODES})
    pairs['guard_vs_geometry'] = ('geometry','guard')
    rows = []
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            u = cv+.5*(old.slate.path_for(cv,b,z['anchor'])-cv)
            i,j = z['old'],z['new']
            p = model.incoming_path(cv[j],ols[j],model.peak(u[i]-cv[i]))
            paths = {base.CV:cv,base.OLS:ols,'half_anchor':u,
                     **{k:model.apply(cv,u,i,j,p,z[k]) for k in study.MODES}}
            paths['calibrated'] = cv if c['selected'] is None else model.apply(cv,u,i,j,p,z['calibrated'])
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            z = {k:a[ix] for k,a in paths.items()}
            r = scorer.score_recording(z,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k:prior.study.motion.full_masses(z[l],z[r],cv[ix],y,v,w) for k,(l,r) in pairs.items()}
            s, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(s),seed=seed,comparisons=terms))
    return rows


def build(path):
    f,r = [study.OUT/k for k in ('freeze.json','readout.json')]
    freeze,main = map(old.read,(f,r))
    if (main['freeze']!=old.sha(f) or main['a_labels_opened'] or main['independent_confirmation']
            or not main['original_2pct']):
        raise ValueError('Require verified development-only readout')
    _,donor,_ = base.register()
    rows = collect(freeze['cells'],base.fit_population(lambda **v:None),donor)
    roster = {x['recording']:x['source'] for x in rows}
    if prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)!=main['summary']:
        raise ValueError('Policy table did not reproduce')
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(schema_version=1,roster={rec[k]:src[v] for k,v in roster.items()},seeds=list(seeds.values()),
        recording_studies={'exchange':pack.trim(rows,rec,src,seeds)},
        joint_cells={k:{str(i):{} for i in seeds.values()} for k in src.values()})
    expected = dict(exchange=dict(main['summary'],seeds=list(seeds.values())),joint_pair_contrasts={})
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)')!=1:
        raise ValueError('Unexpected replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)')+'''
root = Path(__file__).resolve().parent
verify_manifest(root)
actual = recompute(read(root/'statistics.json'))
if actual!=read(root/'expected.json'):
    raise ValueError('The exchange policy table did not replay')
print('Exact anonymous replay: matched exchange risks and utility, 72 views.')
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py. Replays scalar policy scores, not training or capacity.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    print(old.sha(path),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_matched_exchange.zip')
    build(p.parse_args().out)
