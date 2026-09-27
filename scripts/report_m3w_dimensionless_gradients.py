"""All prespecified training-log points, not held-out forecast outcomes."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_refit as run
import numpy as np


def main():
    reg=json.loads((run.PUBLIC/'registration.json').read_text())
    threshold=reg['training']['gradient_clip']
    rows=[]
    for ref in reg['controls']:
        p=run.PRIVATE/'dimensionless'/ref['key']/'complete.json'
        assert run.artifact(ROOT/ref['training']['path'])==ref['training']
        new=json.loads(p.read_text())
        old=json.loads((ROOT/ref['training']['path']).read_text())
        assert new['sampling_exact_vs_grouped'] and new['identity']['registration']==run.artifact(run.PUBLIC/'registration.json')
        a,b=new['fit']['losses'],old['fit']['losses']
        assert [r['step'] for r in a]==[r['step'] for r in b]
        assert a[0]['loss']==b[0]['loss']
        for arm,record in [('dimensionless',new),('grouped',old)]:
            log=record['fit']['losses']
            grad=np.array([r['gradient_norm'] for r in log])
            rows.append(dict(trial=ref['key'],arm=arm,logged_updates=len(log),
                gradient_norm_quantile_levels=[0,.5,.9,1],
                gradient_norm_quantiles=np.quantile(grad,[0,.5,.9,1]).tolist(),
                logged_updates_above_clip=int((grad>threshold).sum()),
                logged_fraction_above_clip=float((grad>threshold).mean()),
                initial_logged_loss=log[0]['loss'],final_logged_loss=log[-1]['loss']))
    doc=dict(result_source='fresh_training_log_comparison',rows=rows,gradient_clip=threshold,
        every_update_recorded=False,held_out_outcomes_scored=False,
        causal_explanation_proven=False,independent_confirmation=False)
    run.immutable_json(run.PUBLIC/'gradient_diagnostic.json',doc)
    lines=['# Logged Gradient Diagnostic','',
        'Same logged steps, initial loss, sample draws and clipping threshold. All nine trials retained.',
        'Logs sample every50updates plus the initial step, not all4,000 gradients.',
        'This comparison cannot isolate optimization changes from coordinate-unit invariance.',
        'Lower training loss or fewer clipped logged gradients is not held-out forecast improvement.','',
        '| Trial | Arm | Logged steps | Above clipping threshold | Median gradient norm | Final logged loss |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['trial']} | {r['arm']} | {r['logged_updates']} | {r['logged_updates_above_clip']} | "+
            f"{r['gradient_norm_quantiles'][1]:.6f} | {r['final_logged_loss']:.6f} |")
    (run.PUBLIC/'gradient_diagnostic.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(trials=9,logged_views=len(rows),held_out_scored=False)))


if __name__=='__main__':
    main()
