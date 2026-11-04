"""Read-only own-source and matched-optimizer audit of completed joint-harm fits."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_harm as run


def audit():
    source, base = run.source, run.api.base
    registration = source.read(run.REG)
    for path, digest in registration['bindings'].items():
        if source.sha(ROOT / path) != digest: raise ValueError('Changed registered definition')
    old = {}
    for ref in source.read(run.old.PUBLIC / 'training_freeze.json')['fits']:
        r = source.read(source.checked(ref))
        if r['replica'] == 0: old[r['job'], r['part']] = r
    pairs, rows = {}, []
    for path in sorted((run.PUBLIC / 'fits').glob('*.json')):
        r = source.read(path); state = base.read_checkpoint(source.checked(r['checkpoint']))
        parent = base.read_checkpoint(source.checked(old[r['job'], r['part']]['checkpoint']))
        if state['step'] != 2000 or state['parameters'] != 28489: raise ValueError('Incomplete or changed model')
        for name in ('fit_ids', 'source_sites', 'preprocess', 'sampler', 'draw_hash', 'settings', 'seed'):
            base.exact(state[name], parent[name])
        if len(state['optimizer']['state']) != 6: raise ValueError('All six parameter tensors must be trained')
        for parameter in state['optimizer']['state'].values():
            if int(parameter['step']) != 2000: raise ValueError('Incomplete optimizer steps')
        if not all(source.torch.isfinite(v).all() for v in state['model'].values()):
            raise ValueError('Nonfinite fitted weights')
        key = (r['job'], r['part']); pair = pairs.setdefault(key, {})
        pair[r['arm']] = state
        rows.append(dict(receipt=source.artifact(path), own_source_scope_and_normalizers_exact=True,
            all_optimizer_parameters_2000_steps=True))
    complete = 0
    for pair in pairs.values():
        if set(pair) != set(run.api.ARMS): continue
        a, b = (pair[k] for k in run.api.ARMS)
        for key in ('initial_model', 'preprocess', 'fit_ids', 'sampler', 'draw_hash', 'torch_rng'):
            base.exact(a[key], b[key])
        if all(source.torch.equal(a['model'][k], b['model'][k]) for k in a['model']):
            raise ValueError('Different objectives did not change any model tensor')
        complete += 1
    return dict(completed_heads_audited=len(rows), complete_matched_pairs=complete,
        full108_complete=len(rows) == 108 and complete == 54, rows=rows,
        read_only_checkpoint_audit=True, validation_not_read=True, scientific_success_not_implied=True)


if __name__ == '__main__':
    result = audit()
    if result['full108_complete']:
        run.source.once(run.PUBLIC / 'fit_audit.json', run.source.stable(result))
    print(run.source.json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
