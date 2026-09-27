"""Post-hoc descriptive mass check, never a model/endpoint selection rule."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_regime_transport as run
import numpy as np


def main():
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    out = dict(status='post_hoc_descriptive_only', new_fit=False, policy_changed=False,
        readout=run.artifact(run.PUBLIC/'readout.json'), families={})
    for family in ('full', 'motion_only'):
        selected = [r for r in rows if r['pair'] == family]
        cells = {}
        for cell in run.method.CELLS:
            records = []
            for row in selected:
                replicas = [r['metrics']['outer_cut3'] for r in row['replicas']]
                if cell == 'three_cut3':
                    for mode in ('raw', 'scaled'):
                        assert all(r[cell+'_'+mode] == replicas[0][cell+'_'+mode] for r in replicas)
                    replicas = replicas[:1]
                records.extend(replicas)
            values = {}
            for mode in ('raw', 'scaled'):
                coverage = np.array([r[cell+'_'+mode]['envelope_positive']['harm_coverage'] for r in records], float)
                assert np.isfinite(coverage).all() and (coverage > 0).all()
                values[mode] = dict(median_mass_ratio=float(np.median(coverage)),
                    below_one_count=int((coverage < 1).sum()), records=len(coverage))
            cells[cell] = dict(outer_views=len(selected), observations_are_dependent=True,
                native_three_site_replicas_deduplicated=cell == 'three_cut3', values=values)
        out['families'][family] = cells
    run.immutable_json(run.PUBLIC/'mass_diagnostic.json', out)
    lines = ['# Harm-Mass Diagnostic', '',
        'Post-hoc descriptive summary of already frozen/scored predictions. No fitting, threshold or endpoint change.',
        'Mass ratio is predicted/observed easy harm, not conformal coverage or physical safety.',
        'Three_cut3 is identical across omitted-site replicas and is deduplicated. Other cells retain three dependent replicas per outer view.',
        'These counts are not independent samples; inferential intervals remain the registered locality bootstrap.', '',
        '| Family/cell | Views/records | Raw median ratio | Scaled median ratio | Raw/scaled below one |',
        '|---|---|---:|---:|---|']
    for family, cells in out['families'].items():
        for cell, doc in cells.items():
            a, b = doc['values']['raw'], doc['values']['scaled']
            lines.append(f"| {family}/{cell} | {doc['outer_views']}/{a['records']} | {a['median_mass_ratio']:.6f} | {b['median_mass_ratio']:.6f} | {a['below_one_count']}/{b['below_one_count']} |")
    lines += ['', 'Obs8/pred12 native steps; detector pixels. No independent confirmation or new deployment claim.',
        'This describes mass underestimation after frozen shrinkage; it does not isolate a universal failure cause.']
    (run.PUBLIC/'mass_diagnostic.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
