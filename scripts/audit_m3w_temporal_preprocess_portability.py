"""Measure TRAIN preprocessing differences across runtimes, without fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def compare(reference, recomputed):
    import numpy as np
    rows = []
    if set(reference) != set(recomputed):
        raise ValueError('Preprocessing keys differ')
    for key, a in reference.items():
        b = recomputed[key]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Preprocessing shape/dtype changed: '+key)
            if a.dtype.kind not in 'fc':
                rows.append(dict(field=key, exact=bool(np.array_equal(a,b)), numeric=False))
                continue
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            a, b = np.asarray(a), np.asarray(b)
        else:
            rows.append(dict(field=key, exact=bool(a==b), numeric=False)); continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite preprocessing: '+key)
        x, y = a.astype(np.float64), b.astype(np.float64)
        diff = np.abs(x-y)
        denominator = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.finfo(np.float64).tiny)
        rows.append(dict(field=key, exact=bool(np.array_equal(a,b)), numeric=True,
            dtype=str(a.dtype), elements=int(a.size), changed=int(np.count_nonzero(a!=b)),
            max_absolute_difference=float(diff.max(initial=0)),
            max_relative_difference=float((diff/denominator).max(initial=0)),
            reference_max_absolute=float(np.abs(x).max(initial=0))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Allocated compute required before numerical imports'
    import numpy as np
    from scripts import train_m3w_temporal_auxiliary_portable as port
    root = args.root.resolve()
    assert root.parent == Path('/users/k24101830/m3w').resolve() and root.name == port.run.NAME
    assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
    manifest = json.loads((root/'train_input_manifest.json').read_text())
    assert manifest['input_role']=='source_TRAIN_only' and not manifest['validation_rows_transferred']
    assert not manifest['independent_roles_read']
    for rel,h in manifest['code_bindings'].items():
        assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==h
    port.run.api.torch.set_num_threads(4); port.run.api.torch.set_num_interop_threads(1)
    refs = {}
    for ref in manifest['heads']:
        if ref['group'] in refs: assert refs[ref['group']]['sha256']==ref['sha256']
        else: refs[ref['group']]=ref
    assert len(refs)==24 and len(manifest['heads'])==72
    rows=[]; started=time.monotonic()
    for ref in refs.values():
        path=root/'inputs'/(ref['group']+'.npz');raw=path.read_bytes()
        assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
        x,env,y,series,sites,rec,frames,pr=port.unpack(raw)
        fresh=port.run.api.core.preprocess(x,env,y,sites,rec,frames,training_site=pr['training_site'])
        rows.append(dict(group=ref['group'],train_rows=len(x),fields=compare(pr,fresh),
            source_packet_sha256=ref['sha256']))
        print(json.dumps(dict(state='TRAIN_preprocess_compared',group=ref['group'],
            fields_not_exact=[r for r in rows[-1]['fields'] if not r['exact']])),flush=True)
    out=dict(result_source='fresh_run_TRAIN_only_preprocessing_diagnostic',
        job_id=os.environ['SLURM_JOB_ID'],numpy=np.__version__,torch=port.run.api.torch.__version__,
        groups=rows,unique_packets=len(rows),seconds=time.monotonic()-started,
        optimizer_updates=0,validation_read=False,independent_roles_read=False,
        original_preprocessing_authority_preserved=True,comparison_tolerance_changed=False,
        scientific_training_executed=False,diagnostic_only=True)
    with (root/'preprocess_portability_diagnostic.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='groups'}),flush=True)


if __name__=='__main__': main()
