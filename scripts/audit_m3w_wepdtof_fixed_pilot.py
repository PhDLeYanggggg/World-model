"""Validate registered fixed-count execution on the exposed camera only."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_readout as exp

study, base, np, torch = exp.study,exp.base,exp.np,exp.torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key,reg,cells = exp.register()
    nets = study.models(reg,cells)
    pilot = study.read(study.OUT/'pilot_freeze.json')
    f = study.read(base.OUT/'rapid_freeze.json')
    windows = []
    for row in f['windows']:
        p = exp.study.camera.detector.PRIVATE/f"window_{row['start']:06}.json"
        if base.sha(p)!=row['cache_sha256']:
            raise ValueError('Pilot paths changed')
        windows.append(study.read(p))
    h,q,ids = study.camera_arrays(windows,pilot=True)
    pred = study.predict(h,q,nets)
    result = {}
    y,v,m,_ = study.outcomes(windows,h,q,ids,study.references(base.PILOT))
    for s,(net,_) in nets.items():
        z = pred[s]
        p = exp.method.bank(z['cv'],z['ols4'],z['spline'],q)
        x,a,l,_ = study.costs.features(h,z['cv'],z['ols4'],p)
        with torch.inference_mode():
            est = net(torch.from_numpy(x)).numpy()*a[:,None,None]
        z.update(exp.fixed(est,l,q,h,z['cv'],p))
        r = exp.augment_metrics(z,y,v,q,base.PILOT,pilot['cut_px'])
        result[str(s)] = {k:r['arms'][k] for k in exp.ACTIVE}
    base.immutable(study.OUT/'fixed_pilot_readout.json',dict(evaluation_registration=key,
        matched=int(m.sum()),results=result,external_labels_read=False,no_tuning=True,original_2pct=True))
    print({s:{k:(v['risk']['selected'],v['risk']['supported'],v['lower_vs_ols4']) for k,v in r.items()}
           for s,r in result.items()},flush=True)


if __name__=='__main__':
    main()
