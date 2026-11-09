"""Check held curve calibration and separate observed risk from missing debit."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_amplitude_response as study

old,base,prior,np,torch,model = study.old,study.base,study.prior,study.np,study.torch,study.model
FIELDS = ('dual_gain','harm_mass','signed_easy','reference_mass','easy_brier')


def load(fit,width):
    z = torch.load(old.checked(fit['artifact']),map_location='cpu',weights_only=True)
    if z['identity']!=fit['identity'] or z['step']!=study.STEPS:
        raise ValueError('Frozen response checkpoint changed')
    net = prior.model.head(width,model.WIDTH)
    net.load_state_dict(z['net'])
    return net.eval()


@torch.inference_mode()
def accuracy(cells,d):
    cells_out = {}
    for cell in cells:
        ident = cell['identity']
        _,h = base.subset(d,ident['held'])
        cv,ols,b = study.analytic(h['history'])
        past,s = study.parent.context.context(h)
        nets = [load(f,past.shape[1]+37) for f in cell['fits']]
        legal = (b!=cv[:,None]).any((2,3))
        for i,site in enumerate(sorted(set(h['sites']))):
            rng = np.random.default_rng(7189+10*ident['fold']+i)
            rows = np.flatnonzero((h['sites']==site)&legal.any(1))
            ix = rng.choice(rows,min(len(rows),1024),replace=False)
            j = np.where(legal[ix],rng.random(legal[ix].shape),-1.).argmax(1)
            path = b[ix,j]
            f = torch.from_numpy(study.parent.context.model.action_features(past[ix],cv[ix],ols[ix],path,s[ix]))
            y,m,e = model.targets(cv[ix],ols[ix],path,h['target'][ix],h['valid'][ix],s[ix],cell['cut'][0])
            values = []
            for k,net in enumerate(nets):
                p,r,z = model.response(net(f),k==1)
                err = np.abs(p.numpy()-y)
                values.append([err[...,:2].mean(),err[...,2:4].mean(),err[...,4].mean(),
                               np.abs(r.numpy()-m).mean(),np.square(z.sigmoid().numpy()-e).mean()])
            cells_out[str(site),ident['seed']] = np.asarray(values,float)
    sites = sorted({k[0] for k in cells_out})
    if len(sites)!=9 or len(cells_out)!=27:
        raise ValueError('Expected all held source/seed diagnostic cells')
    z = np.asarray([[cells_out[s,k] for k in base.SEEDS] for s in sites]).mean(1)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    d = z[:,0]-z[:,1]
    return {k:dict(direct=float(z[:,0,i].mean()),shape=float(z[:,1,i].mean()),
        direct_minus_shape=float(d[:,i].mean()),interval95=np.quantile(d[:,i][draw].mean(1),[.025,.975]).tolist())
        for i,k in enumerate(FIELDS)}


def failures(rows):
    out = {}
    for k in study.ARMS:
        rs = [r['metrics'][k]['risk'] for r in rows]
        known = [all(x[key] is not None and x[key]<=.02 for key in
                    ('known_all_risk','known_easy_risk','known_easy_net_degradation')) for x in rs]
        out[k] = dict(known_only_passes=sum(known),original_passes=sum(x['supported'] for x in rs),
            missing_debit_rejections=sum(a and not x['supported'] for a,x in zip(known,rs,strict=True)),
            wholly_unknown_edits=sum(x['unknown_selected'] for x in rs))
    return out


def main():
    torch.set_num_threads(4)
    r,f = [study.OUT/k for k in ('readout.json','freeze.json')]
    a,b = map(old.read,(r,f))
    if (a['freeze']!=old.sha(f) or a['a_labels_opened'] or a['independent_confirmation'] or not a['original_2pct']):
        raise ValueError('Expected fixed development results')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    rows = study.evaluate(b['cells'],d,donor,lambda **v:None,export_rows=True)
    if prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)!=a['summary']:
        raise ValueError('Diagnostic does not reproduce the original policy table')
    out = dict(parent=old.artifact(r),code=old.artifact(Path(__file__)),accuracy=accuracy(b['cells'],d),
        risk=failures(rows),sample_cap_per_source_seed=1024,source_seed_cells=27,diagnostic_only=True,
        a_labels_opened=False,independent_confirmation=False,original_2pct=True)
    old.once(study.OUT/'response_audit.json',old.stable(out))
    print(out['accuracy'],out['risk'],flush=True)


if __name__=='__main__':
    main()
