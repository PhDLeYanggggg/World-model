"""Locate the specific predicted filter that excludes realized donor actions."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_pooled_coverage as study

np,torch,base,prior,screen = study.np,study.torch,study.base,study.prior,study.screen
read,sha,artifact,once,stable = study.read,study.sha,study.artifact,study.once,study.stable


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = study.OUT
    parent = out/'support_attribution.json'
    a = read(parent)
    score = out/'readout.json'
    if a['parent']['sha256']!=sha(score) or a['independent_confirmation']:
        raise ValueError('Expected fitting-source support evidence')
    f = read(out/'freeze.json')
    if read(score)['freeze']!=sha(out/'freeze.json'):
        raise ValueError('Choice freeze changed')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    counts, candidate_counts = Counter(),Counter()
    scorer = prior.prior.parent.readout
    for old,cell in zip(read(base.OUT/'freeze.json')['cells'],f['cells'],strict=True):
        ident = cell['identity']
        if ident!=old['identity'] or old['fits'][0]['identity']!=dict(ident,head='risk'):
            raise ValueError('Fitting roles changed')
        hi,h = base.subset(d,ident['held'])
        cv,_,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        bank = base.expanded(cv,raw)
        width = prior.study.motion.features(h['history'][:1],'prefix')[0].shape[1]
        net = screen.load_head(old['fits'][0],width+25,8)
        _,risk,prob,legal = screen.predictions(net,cv,bank,h['history'])
        y,v = h['target'],h['valid']
        cut = cell['cut'][0]
        ade,_ = scorer.errors(cv,y,v)
        truth = legal & (v.any(1)&(ade>0.)&(ade<=cut))[:,None]
        truth &= (prior.utility.risk_excess(cv,bank,y,v,cut)<=0.).all(2)
        easy = prob>=.10
        all_ok = risk[...,0]<=0.
        easy_ok = risk[...,1]<=0.
        stages = dict(true_donor=truth,probability_only=truth&easy,predicted_all_only=truth&all_ok,
            predicted_easy_only=truth&easy_ok,both_costs_only=truth&all_ok&easy_ok,
            all_three_same_action=truth&easy&all_ok&easy_ok)
        groups = prior.query_groups(h['query'])
        for k,m in stages.items():
            row = m.any(1)
            counts[k] += sum(bool(row[g].any()) for g in groups)
            candidate_counts[k] += int(m.sum())
        print(dict(fold=ident['fold'],seed=ident['seed']),flush=True)
    if counts['true_donor']!=a['query_donors']['actual']:
        raise ValueError('Known donor query denominator changed')
    once(out/'support_funnel.json',stable(dict(parent=artifact(parent),code=artifact(Path(__file__)),
        query_views=dict(counts),candidate_row_path_views=dict(candidate_counts),
        classification_at_true_donor_action=True,hindsight=True,independent_confirmation=False,
        original_2pct=True)))


if __name__=='__main__':
    main()
