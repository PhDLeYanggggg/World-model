"""Nested source credit transport and unchanged-risk readout."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_causal_record_ledger as prior
from src.world_model import m3w_ledger_credit_transport as model

run, np, nested = prior.run, prior.np, prior.nested
PUBLIC = prior.PUBLIC/'credit_transport'; PRIVATE = prior.PRIVATE/'credit_transport'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ledger_credit_transport.py')
ARMS = ('query','full','half','donor_discount')


def register():
    a,b,c = (prior.PUBLIC/f for f in ('analysis.json','replay_execution.json','freeze.json'))
    if run.read(b)['artifact']['sha256'] != run.sha(a) or not run.read(c)['chronological']:
        raise ValueError('Verified chronological parent required')
    inputs = (a,b,c,prior.REG,Path(prior.__file__),Path(prior.model.__file__),nested.REG,Path(nested.__file__),
              Path(nested.cal.__file__),run.REG,run.PUBLIC/'freeze.json')
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],parents=[run.artifact(p) for p in inputs],
        arms=ARMS,half_retention=.5,donor_quantile=.10,min_donors=3,all_counts_matched=True,
        original_rules_unchanged=True,external_outcomes=False)))


def source_groups(b,frames):
    by_record = {}
    for g in b['groups']:
        r = str(b['rec'][g[0]])
        if not np.all(b['rec'][g]==b['rec'][g[0]]) or not np.all(frames[g]==frames[g[0]]):
            raise ValueError('Current query crossed recording or time')
        by_record.setdefault(r,[]).append(g)
    for groups in by_record.values():
        groups.sort(key=lambda g:frames[g[0]])
        if len({frames[g[0]] for g in groups}) != len(groups):
            raise ValueError('Recording frame must identify a unique query')
    return by_record


def true_excess(b):
    h,r,eh,er = b['actual'].transpose(2,0,1)
    return h-.02*r, eh-.02*er


def donor(b, groups, c):
    u,a,e,ok = nested.predict(b,c); scale = b['scale']; ta,te = true_excess(b)
    rows = []
    for rec,g in groups.items():
        parent = nested.base.select(g,u,a,e,scale,ok)
        counts = [(parent[x]>=0).sum() for x in g]
        ch,_ = prior.model.stream(g,u/scale[:,None],a,e,ok,counts=counts,witness=parent)
        p,y = model.query_totals(g,ch,a,e),model.query_totals(g,ch,ta,te)
        rows.append(dict(recording=rec,**model.earned_credit(p,y)))
    return rows


def fit_seed(d,refs,frames,old,labels,beat):
    seed = old['seed']; cv,paths = run.candidates(d,refs,seed)
    blocks = nested.prepare(d,paths,cv,labels,run.cuts_by_source())
    groups = {s:source_groups(b,frames[b['ix']]) for s,b in blocks.items()}
    inner = {(h,c):run.model.coefficients([b['moments'] for s,b in blocks.items() if s not in (h,c)])
             for i,h in enumerate(blocks) for c in list(blocks)[i+1:]}
    with np.load(run.checked(old['cache']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['global_ids']); names,outer = z['names'],z['coefficients']
    ref = next(v for v in run.read(prior.PUBLIC/'freeze.json')['seeds'] if v['seed']==seed)
    with np.load(run.checked(ref['choices']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['global_ids']); parent = z['choices'][:3]
    choices = np.full((len(ARMS),len(cv)),-1,dtype=np.int16); estimates = []; stats = []
    for held,c in zip(names,outer):
        donors = {s:donor(b,groups[s],inner[tuple(sorted((held,s)))]) for s,b in blocks.items() if s!=held}
        discount = model.calibrate(donors,held)
        estimates.append(dict(source=str(held),donor_sources=list(donors),**discount))
        b = blocks[held]; ix = b['ix']; u,a,e,ok = nested.predict(b,c); value = u/b['scale'][:,None]
        for r,g in groups[held].items():
            count = [(parent[0,ix[k]]>=0).sum() for k in g]; witness = parent[0,ix]
            local = np.concatenate(g)
            for j,alpha in enumerate(([0.,0.],[1.,1.],[.5,.5],discount['alpha'])):
                ch,m = model.stream(g,value,a,e,ok,count,witness,alpha)
                choices[j,ix[local]] = ch[local]
                stats.append(dict(source=str(held),recording=r,arm=ARMS[j],**m))
        beat(state='calibrated_source',seed=seed,source=str(held),retention=discount['alpha'])
    np.testing.assert_array_equal(choices[:2],parent[1:3])
    dest = PRIVATE/f'{seed}.npz'; tmp = dest.with_suffix('.pending.npz')
    np.savez_compressed(tmp,ids=d['global_ids'],choices=choices); tmp.replace(dest)
    return dict(seed=seed,decisions=run.artifact(dest),calibrators=estimates,stats=stats,controls_exact=True)


def fit(d,refs,frames,beat):
    labels = run.old.prior.arrays(('target_eval','valid')); seeds = []
    for old in run.read(run.PUBLIC/'freeze.json')['seeds']:
        mark = PRIVATE/f'{old["seed"]}.json'
        if not mark.exists():
            run.once(mark,dict(registration=run.sha(REG),**fit_seed(d,refs,frames,old,labels,beat)))
        doc = run.read(mark)
        if doc['registration']!=run.sha(REG) or not doc['controls_exact']:
            raise ValueError('Changed seed or fitting lineage')
        run.checked(doc['decisions']); seeds.append(doc)
    run.once(PUBLIC/'freeze.json',dict(seeds=seeds,all_decisions_before_readout=True))


def score(d,refs,frames,beat):
    f = run.read(PUBLIC/'freeze.json'); cuts = run.cuts_by_source(); p = run.parent
    if not f['all_decisions_before_readout'] or len(f['seeds'])!=3:
        raise ValueError('Complete policy freeze required')
    labels = run.old.prior.arrays(('target_eval','valid')); y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    blocks = prior.blocks(d,frames); records = []; credits = []
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(half_vs_full=('full','half',p.CV),donor_vs_full=('full','donor_discount',p.CV),
                 donor_vs_query=('query','donor_discount',p.CV))
    ctrl = {(v['seed'],v['recording']):v['metrics'] for v in run.read(prior.PUBLIC/'analysis.json')['records']}
    old = {v['seed']:v for v in run.read(run.PUBLIC/'freeze.json')['seeds']}
    for cell in f['seeds']:
        seed = cell['seed']; cv,paths = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['decisions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids']); choice = z['choices']
        with np.load(run.checked(old[seed]['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids']); coef = dict(zip(z['names'],z['coefficients']))
        for r,(ix,g) in blocks.items():
            s = str(d['sites'][ix[0]]); cut = cuts[s]; ch = choice[:,ix]
            for j in g:
                n = (ch[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Matched count changed')
            pred = {a:run.model.selected_paths(c,paths[ix],cv[ix]) for a,c in zip(ARMS,ch)}
            v = p.readout.score_recording({p.CV:cv[ix],**pred},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cut['easy'],hard_cut=cut['hard'])
            prior_metrics = ctrl[seed,r]
            if v['metrics']['query'] != prior_metrics['query_matched'] or v['metrics']['full'] != prior_metrics['ledger_matched']:
                raise ValueError('Control scalar replay failed')
            x,scale = run.model.descriptors(d['history'][ix],paths[ix],cv[ix],d['crowd'][ix])
            _,a,e,_ = run.model.decision_costs(x,coef[s],paths[ix],cv[ix],scale)
            radius = np.linalg.norm(paths[ix]-cv[ix,None],axis=-1).max(2)
            act = nested.cal.observed_components(paths[ix],cv[ix],y[ix],valid[ix],radius,cut['easy'])
            ta,te = true_excess(dict(actual=act))
            pp = model.query_totals(g,ch[1],a,e); yy = model.query_totals(g,ch[1],ta,te)
            credits.append(dict(seed=seed,source=s,recording=r,**model.earned_credit(pp,yy),
                borrowed_queries=int((pp>prior.model.EPS).any(1).sum()),
                true_deficit_on_borrow=int(((pp>prior.model.EPS)&(yy>prior.model.EPS)).any(1).sum())))
            records.append(dict(seed=seed,source=s,**v))
        beat(state='scored_seed',seed=seed,views=len(records))
    if len(records)!=102 or len(credits)!=102:
        raise ValueError('All views must remain')
    sites = np.unique(d['sites']); draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    grouped = [[r for r in records if r['source']==s and r['seed']==seed] for s in sites for seed in run.kin.prior.SEEDS]
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sites),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        m = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        comparisons[k] = {f:ci([v[f] for v in m]) for f in ('known_pp','completion_lower_pp')}
    risks = {}
    for a in ARMS:
        views = [r['metrics'][a]['risk'] for r in records]
        risks[a] = dict(passes=sum(v['supported'] for v in views),switches=sum(v['selected'] for v in views),
            easy_over_2=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper']>.02 for v in views),
            undefined_easy=sum(v['completion_easy_risk_upper'] is None for v in views))
    totals = {key:np.sum([r[key] for r in credits],axis=0).tolist() for key in
              ('predicted','realized','earning_queries','positive_observed_on_credit','borrowed_queries','true_deficit_on_borrow')}
    alpha = np.asarray([c['alpha'] for s in f['seeds'] for c in s['calibrators']]); changed = 0
    for s in f['seeds']:
        with np.load(run.checked(s['decisions']),allow_pickle=False) as z:
            ch = z['choices']; changed += int((ch[0]!=ch[3]).sum())
    run.once(PUBLIC/'analysis.json',run.stable(dict(risks=risks,comparisons=comparisons,records=records,
        credit_audit=credits,credit_totals=totals,discount_positive_folds=(alpha>0).sum(0).tolist(),
        donor_changes_vs_query=changed,controls_exact=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase; register()
    if phase=='register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)!=f.read_bytes():
            raise ValueError('Commit registered experiment before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True); start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Preserve seed checkpoints for resume')
            pkt = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',pkt); print(pkt,flush=True)
        beat(state='start'); d = run.population(); refs = run.register(); frames = run.old.prior.arrays(('frames',))['frames']
        if phase=='fit':
            fit(d,refs,frames,beat)
        else:
            score(d,refs,frames,beat)
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=run.artifact(PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json'))))


if __name__=='__main__':
    main()
