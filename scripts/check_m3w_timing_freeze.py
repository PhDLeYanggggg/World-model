"""Verify the actual frozen action agreement without reading target labels."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_timing_capacity as study

run, np = study.run, study.np


def main():
    root = study.PUBLIC
    freeze = root/'freeze.json'
    doc = run.read(freeze)
    if run.read(root/'freeze_execution.json')['artifact']['sha256']!=run.sha(freeze):
        raise ValueError('Changed frozen decisions')
    if not doc['no_development_targets_loaded']:
        raise ValueError('Only pre-scoring frozen decisions are eligible')
    rows = []
    for f in doc['cells']:
        with np.load(run.checked(f['cache']),allow_pickle=False) as z:
            c = z['choices']
            if c.shape!=(3,125540) or len(np.unique(z['ids']))!=125540:
                raise ValueError('Original complete action roster required')
        rows.append(dict(seed=f['identity']['seed'],
                         switches=dict(zip(study.ARMS,(c>=0).sum(1).tolist())),
                         local_vs_direct_changed=int((c[1]!=c[0]).sum()),
                         banked_vs_local_changed=int((c[2]!=c[1]).sum())))
    if sorted(r['seed'] for r in rows)!=[17,29,43] or sum(r['switches']['direct'] for r in rows)!=6952:
        raise ValueError('Original controls changed')
    run.once(root/'frozen_agreement.json',run.stable(dict(
        freeze_sha256=run.sha(freeze),causal_arms=study.ARMS,seeds=rows,target_labels_opened=False)))
    print(rows)


if __name__ == '__main__':
    main()
