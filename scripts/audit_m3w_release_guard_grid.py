"""Label-aware threshold-grid ceiling, never exported as a selected threshold."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_release_guard as study

run,np = study.run,study.np


def main():
    path = study.PUBLIC/'analysis.json'
    old = run.read(path)
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Run the frozen outer evaluation and replay first')
    d,refs = run.population(),run.register()
    g = study.prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
    label = run.old.prior.arrays(('target_eval','valid'))
    y,valid = label['target_eval']-d['origin'][:,None],label['valid']
    cuts = run.cuts_by_source()
    rows = []
    for cell in run.read(study.PUBLIC/'freeze.json')['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        for c in cell['selection']:
            ident = c['identity']
            held = ident['held']
            if held in (*ident['fitting'],*ident['calibration']):
                raise ValueError('Saved outer provenance mismatch')
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                coef,correction,harm = z['baseline_coefficients'],z['residual_coefficients'],z['guard_coefficients']
            table = study.calibrate(d,g,(held,),cv,bank,coef,correction,harm,y,valid,cuts)
            for arm in study.guard.GRIDS:
                candidates = table[arm]['candidates']
                selected = c['selection'][arm]['selected']
                value = candidates[selected]['lower_pp']
                if value is None or not candidates[0]['preserved']:
                    raise ValueError('Expected estimable lower and preserved base')
                best = max(q['lower_pp'] for q in candidates if q['preserved'] and q['lower_pp'] is not None)
                rows.append(dict(seed=seed,source=held,arm=arm,actual_lower=value,
                    chosen_preserved=candidates[selected]['preserved'],best_preserved_lower=best))
    if len(rows)!=84:
        raise ValueError('Require 42 held-source/seed cells for each score')
    sources = sorted({r['source'] for r in rows})
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    result = {}
    for arm in study.guard.GRIDS:
        a = [r for r in rows if r['arm']==arm]
        v = np.asarray([np.mean([r['best_preserved_lower'] for r in a if r['source']==s]) for s in sources])
        value = float(np.mean([r['actual_lower'] for r in a]))
        np.testing.assert_allclose(value,old['comparisons'][f'{arm}_vs_base']['completion_lower_pp']['point'],atol=1e-12)
        result[arm] = dict(chosen_fails_outer_preservation=sum(not r['chosen_preserved'] for r in a),
            cells_with_positive_preserving_option=sum(r['best_preserved_lower']>0. for r in a),
            best_grid_lower_vs_base_pp=dict(point=float(v.mean()),
                interval95=np.quantile(v[draw].mean(1),[.025,.975]).tolist()))
    run.once(study.PUBLIC/'grid_ceiling.json',dict(source=run.artifact(path),arms=result,
        diagnostic_only=True,thresholds_not_exported=True,does_not_certify_original_full_gate=True,
        independent_outcomes_opened=False))


if __name__=='__main__':
    main()
