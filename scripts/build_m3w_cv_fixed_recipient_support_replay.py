"""Export anonymous scalar replay and count actual path changes."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import train_m3w_cv_fixed_recipient_support as study

old,base,prior,np = study.old,study.base,study.prior,study.np


def collect(cells,d,donor):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in study.ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(pooled_vs_uniform=('uniform','pooled'),consensus_vs_uniform=('uniform','consensus'))
    rows,counts = [],dict.fromkeys(study.MODES,0)
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = z['anchor']
            p = {k:study.paths(cv,b,a,z[k],.5) for k in study.MODES}
            p['calibrated'] = (cv if c['selected'] is None else
                study.paths(cv,b,a,z['calibrated'],c['selected']['alpha']))
        p.update({base.CV:cv,base.OLS:ols,'frozen':old.slate.path_for(cv,b,a)})
        for k in study.MODES:
            counts[k] += int((np.abs(p[k]-p['uniform']).max((1,2))>1e-6).sum())
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            z = {k:p[k][ix] for k in p}
            r = scorer.score_recording(z,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k:prior.study.motion.full_masses(z[l],z[r],cv[ix],y,v,w) for k,(l,r) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=terms))
    return rows,counts


def build(path):
    f,r = [study.OUT/k for k in ('freeze.json','readout.json')]
    freeze,readout = map(old.read,(f,r))
    if (readout['freeze']!=old.sha(f) or readout['a_labels_opened']
            or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Require frozen development-only tables')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    rows,counts = collect(freeze['cells'],d,donor)
    roster = {x['recording']:x['source'] for x in rows}
    if prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)!=readout['summary']:
        raise ValueError('Policy table did not reproduce')
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(schema_version=1,roster={rec[k]:src[v] for k,v in roster.items()},seeds=list(seeds.values()),
        recording_studies={'equal_peak':pack.trim(rows,rec,src,seeds)},
        joint_cells={k:{str(i):{} for i in seeds.values()} for k in src.values()})
    expected = {'equal_peak':dict(readout['summary'],seeds=list(seeds.values())),'joint_pair_contrasts':{}}
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)')!=1:
        raise ValueError('Unexpected reviewer replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)')+'''
root = Path(__file__).resolve().parent
verify_manifest(root)
actual = recompute(read(root/'statistics.json'))
if actual!=read(root/'expected.json'):
    raise ValueError('Scalar readout did not replay')
print('Exact anonymous replay: equal-peak table, 72 views.')
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py. Replays scalar policy risks and utility, not training or geometry.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    old.once(study.OUT/'replay_receipt.json',old.stable(dict(parent=old.artifact(r),
        code=old.artifact(Path(__file__)),candidate_nondefault=readout['changed_directions'],
        actual_path_changes=counts,coordinate_difference_threshold=1e-6,
        anonymous_sha256=old.sha(path),original_2pct=True,independent_confirmation=False)))
    print(old.sha(path),counts,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_fixed_recipient_support.zip')
    build(p.parse_args().out)
