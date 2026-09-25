"""Post-readout accounting only; no threshold search, fitting or selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selection_readout as run
from scripts import build_m3w_european_selection_data as adapter
from scripts.report_m3w_european_floor_relative import dump
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.evaluation.m3w_native_metrics import native_errors


def main():
    cfg, bank, identity = run.registration()
    data = adapter.load(run, identity); labels = adapter.load(run, identity, labels=True)
    rows = [json.loads(p.read_text()) for p in sorted((run.PUBLIC/'groups').glob('*.json'))]
    assert len(rows) == 36
    by_event = {}
    for event in ('all', 'easy'):
        rr = [r for r in rows if r['metadata']['event'] == event]
        out = {}
        for policy in ('old_stop', 'add_only', 'incumbent_reference', 'ridge_incumbent'):
            ss = [r['summary'][policy] for r in rr]
            ratios = [v['realized_event_harm_ratio'] for r in rr
                for v in r['views'][policy]['risk_reliability'].values() if v['realized_event_harm_ratio'] is not None]
            out[policy] = dict(views=len(rr), easy_pass=sum(s['easy_pass'] for s in ss),
                gain_range=[min(s['gain_vs_incumbent'] for s in ss), max(s['gain_vs_incumbent'] for s in ss)],
                gain_vs_training_selected_range=[min(s['gain_vs_training_selected'] for s in ss), max(s['gain_vs_training_selected'] for s in ss)],
                positive_CI=sum(s['CI'] is not None and s['CI'][0] > 0 for s in ss),
                worst_easy_degradation=max(s['worst_easy_degradation'] for s in ss),
                positive_harm_above02=sum(v > .02 for v in ratios), risk_locality_views=len(ratios),
                maximum_realized_positive_harm_ratio=max(ratios))
        by_event[event] = out
    worst = min((v['gain_percent'], r['group'], site) for r in rows
        for site, v in r['views']['add_only']['ADE_vs_CV']['easy']['by_scene'].items() if v['gain_percent'] is not None)
    _, name, site = worst; group = bank['groups'][name]
    receipt = json.loads((run.PRIVATE/'decisions'/(name+'.json')).read_text())
    with np.load(run.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
    neural = np.load(ROOT/receipt['artifacts']['neural']['path'], allow_pickle=False, mmap_mode='r')
    def ade(p): return native_errors(p, labels['target_eval'], labels['valid'], np.ones(len(p)))[0]
    cv = ade(baseline_numpy(data['history'], 1))
    floor = np.where(a['floor_bit'][:, None, None], baseline_numpy(data['history'], 3), baseline_numpy(data['history'], 1))
    da = ade(floor); na = ade(neural.astype(float)+data['origin'][:, None])
    old, added = np.where(a['old_stop'], na, da), np.where(a['add_only'], na, da)
    easy = (data['sites'] == site)&(cv > 0)&(cv <= group['easy_cut'])
    new = a['add_only'] & ~a['old_stop']; use = easy & new; delta = added-old
    den = float(cv[easy].sum()); harm = float(np.maximum(delta[use], 0).sum()); gain = float(np.maximum(-delta[use], 0).sum())
    r = dict(group=name, site=site, easy_rows=int(easy.sum()), new_easy_interventions=int(use.sum()),
        beneficial_additions=int((use & (delta < 0)).sum()), harmful_additions=int((use & (delta > 0)).sum()),
        neutral_additions=int((use & (delta == 0)).sum()), cv_error_sum=den,
        added_harm=harm, added_benefit=gain, net_harm_pp=100*(harm-gain)/den,
        old_easy_degradation_pp=100*(old[easy].sum()/den-1),
        new_easy_degradation_pp=100*(added[easy].sum()/den-1),
        easy_added_positive_harm_over_old_error=harm/float(old[easy].sum()),
        mean_predicted_added_harm=float(a['incumbent_reference__utility'][use, 1].mean()),
        mean_realized_added_harm=float(np.maximum(delta[use], 0).mean()))
    np.testing.assert_allclose(r['old_easy_degradation_pp']+r['net_harm_pp'], r['new_easy_degradation_pp'], atol=1e-12)
    dump(run.PUBLIC/'diagnostic_accounting.json', dict(result_source='fresh_run_post_readout_accounting',
        by_event=by_event, worst_add_only_easy=r, policy_changed=False, no_new_fitting=True,
        caution='Repeated locality views; event cohorts have different masks, no formal calibration claim'))
    print(json.dumps(dict(by_event=by_event, worst=r), indent=2))


if __name__ == '__main__': main()
