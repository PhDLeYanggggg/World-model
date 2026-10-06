"""Allocated TRAIN-only diagnosis of the historical exact-control failure."""
import argparse
import json
import os
from pathlib import Path
import platform

from scripts import train_m3w_easy_harm_deviance as run


def compare(left, right, exact):
    import torch

    result = {}
    for name in ('initial_model', 'preprocess', 'settings', 'input_hashes',
                 'seed', 'step', 'sampler_rng', 'torch_rng', 'draw_hash',
                 'row_draws', 'queries', 'model', 'optimizer'):
        try:
            exact(left[name], right[name])
            result[name] = {'exact': True}
        except (AssertionError, ValueError):
            result[name] = {'exact': False}
    tensor_deltas = []

    def walk(a, b, path):
        if isinstance(a, dict):
            if set(a) != set(b):
                tensor_deltas.append(dict(path=path, structure_mismatch=True))
            else:
                for k in a:
                    walk(a[k], b[k], path+'/'+str(k))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                tensor_deltas.append(dict(path=path, structure_mismatch=True))
            else:
                for k, (aa, bb) in enumerate(zip(a, b)):
                    walk(aa, bb, path+'/'+str(k))
        elif isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                tensor_deltas.append(dict(path=path, structure_mismatch=True))
            elif not torch.equal(a, b):
                delta = (a.double()-b.double()).abs()
                tensor_deltas.append(dict(path=path, different=int((a != b).sum()),
                    elements=a.numel(), max_abs=float(delta.max()),
                    rms=float(delta.square().mean().sqrt()),
                    finite=bool(torch.isfinite(delta).all())))

    walk(left['model'], right['model'], 'model')
    walk(left['optimizer'], right['optimizer'], 'optimizer')
    result['tensor_deltas'] = tensor_deltas
    a = {r['step']: r['monitor']['primary_total'] for r in left['trace']}
    b = {r['step']: r['monitor']['primary_total'] for r in right['trace']}
    result['monitor_deltas'] = [dict(step=s, left=a[s], right=b[s], delta=a[s]-b[s])
                                for s in sorted(set(a) & set(b))]
    result['all_control_fields_exact'] = all(result[k]['exact'] for k in result if isinstance(result[k], dict))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    root, source, cfg, reg, manifest = run.verify(args.root)
    binding = json.loads((root/'control_diagnostic_registration.json').read_text())
    assert run.sha(__file__) == binding['script_sha256']
    assert run.sha(root/'registration.json') == binding['training_registration_sha256']
    from src.world_model import m3w_easy_harm_deviance_training as api
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    ref = manifest['heads'][0]
    data = run.packet(source, ref)
    home = root/run.PRIVATE/'control_diagnostic_v1'
    home.mkdir(parents=True, exist_ok=True)
    kw = dict(settings=cfg['head_training'], seed=ref['identity']['seed'],
              identity=ref['identity'], heartbeat=lambda **v: print(json.dumps(v), flush=True),
              checkpoint_guard=lambda: run.quota(root, source, cfg))
    states = {}
    with frozen_preprocess(api.core, data[-1]):
        path = home/'old_direct.pt.gz'
        states['old_direct'] = api.parent.fit(*data, arm='none', path=path,
                                             resume=path.exists(), **kw)
        path = home/'new_direct.pt.gz'
        states['new_direct'] = api.fit(*data, arm='quadratic', path=path,
            experiment_sha256=run.sha(root/'registration.json'), resume=path.exists(), **kw)
        path = home/'old_from_new_pilot.pt.gz'
        if not path.exists():
            run.quota(root, source, cfg)
            api.core.save_checkpoint(path, api.core.read_checkpoint(root/run.PRIVATE/'original_control_pilot.pt.gz'))
        states['old_from_new_pilot'] = api.parent.fit(*data, arm='none', path=path,
                                                    resume=True, **kw)
    states['failed_new'] = api.core.read_checkpoint(root/run.PRIVATE/'heads'/run.key(ref, 'quadratic')/'checkpoint.pt.gz')
    refs = json.loads((source/run.OLD_PUBLIC/'training_freeze.json').read_text())['fits']
    for item in refs:
        doc = json.loads((source/item['path']).read_text())
        if doc['arm'] == 'none' and doc['identity'] == ref['identity']:
            cp = source/doc['checkpoint']['path']
            assert run.sha(cp) == doc['checkpoint']['sha256']
            states['historical_old'] = api.core.read_checkpoint(cp)
            break
    assert 'historical_old' in states
    pairs = [('old_direct', 'new_direct'), ('old_direct', 'historical_old'),
             ('new_direct', 'failed_new'), ('old_from_new_pilot', 'failed_new'),
             ('old_direct', 'old_from_new_pilot')]
    cpu = [s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
           if s.startswith('model name')]
    out = dict(result_source='fresh_run_TRAIN_only_execution_diagnostic', job_id=os.environ['SLURM_JOB_ID'],
        node=platform.node(), cpu_model=sorted(set(cpu)), torch=api.torch.__version__,
        cpu_threads=4, workers=0, script_sha256=run.sha(__file__),
        comparisons={a+'_vs_'+b: compare(states[a], states[b], api.core.exact) for a,b in pairs},
        checkpoints={k: dict(path=str(v.relative_to(root)), sha256=run.sha(v), bytes=v.stat().st_size)
                     for k,v in ((q, home/(q+'.pt.gz')) for q in ('old_direct', 'new_direct', 'old_from_new_pilot'))},
        independent_roles_read=False, validation_scored=False,
        acceptance_rule_relaxed=False, scientific_lift_established=False)
    run.once(root/run.PUBLIC/'control_diagnostic_v1.json', out)
    print(json.dumps(dict(state='diagnostic_complete',
        exact={k:v['all_control_fields_exact'] for k,v in out['comparisons'].items()})), flush=True)


if __name__ == '__main__':
    main()
