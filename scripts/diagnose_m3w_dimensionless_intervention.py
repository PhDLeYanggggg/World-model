"""Post-freeze source risk diagnostics; no policy fitting or threshold selection."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_dimensionless_intervention as run
from src.evaluation.m3w_native_metrics import native_errors


def ratio(a, b):
    return float(np.sum(a)/np.sum(b)) if np.sum(b)>0 else None


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    cfg, data, jobs, identity, _ = run.load()
    run.committed(run.PUBLIC/'decision_freeze.json')
    rows = []
    for j in jobs:
        bank_ids, cvp, bank = run.candidates(j, data)
        for controller in range(3):
            if controller == j['old_identity']['fold']: continue
            for candidate, p in bank.items():
                name = run.group_name(j, controller, candidate)
                ids, s, _, _ = run.scores(name, identity)
                pos = np.searchsorted(bank_ids, ids)
                cv = data['baseline_ade'][ids,1]
                cost = native_errors(p[pos].astype(float)+data['origin'][ids,None], data['target_eval'][ids],
                                     data['valid'][ids], np.ones(len(ids)))[0]
                moving = np.linalg.norm(data['history'][ids,-1]-data['history'][ids,-2], axis=1)>0
                u, a, e = (s[k] for k in ('utility','all','easy'))
                reasons = np.zeros(len(ids), np.int8)
                reasons[moving] = 1
                reasons[moving & (u[:,0]>u[:,1])] = 2
                reasons[(reasons==2) & (a[:,1]<=cfg['risk_budget']*a[:,0])] = 3
                reasons[(reasons==3) & (e[:,1]<=cfg['risk_budget']*e[:,0])] = 4
                with np.load(run.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ids)
                    np.testing.assert_array_equal(z['point'], reasons==4)
                for site in sorted(set(data['sites'][ids])):
                    at=data['sites'][ids]==site; known=at & np.isfinite(cv)
                    selected=known & (reasons==4); easy=known & (cv>0) & (cv<=j['design']['easy_cut'])
                    ez=selected & easy; zero=known & (cv==0)
                    harm=np.maximum(cost-cv,0); benefit=np.maximum(cv-cost,0)
                    names=('stationary_last_step','nonpositive_predicted_gain','all_risk_veto','easy_risk_veto','selected')
                    rows.append(dict(group=name, candidate=candidate, site=str(site),
                        reasons={k:int((at & (reasons==i)).sum()) for i,k in enumerate(names)},
                        known_rows=int(known.sum()), selected_rows=int(selected.sum()),
                        actual_positive_harm_ratio_all=ratio(np.where(selected,harm,0)[known],cv[known]),
                        actual_positive_harm_ratio_easy=ratio(np.where(selected,harm,0)[easy],cv[easy]),
                        selected_predicted_all_ratio=ratio(a[selected,1],a[selected,0]),
                        selected_actual_all_ratio=ratio(harm[selected],cv[selected]),
                        selected_predicted_easy_ratio=ratio(e[selected,1],e[selected,0]),
                        selected_actual_easy_ratio=ratio(harm[ez],cv[ez]),
                        selected_predicted_benefit_sum=float(u[selected,0].sum()),
                        selected_actual_benefit_sum=float(benefit[selected].sum()),
                        selected_predicted_harm_sum=float(a[selected,1].sum()),
                        selected_actual_harm_sum=float(harm[selected].sum()),
                        zero_reference_rows=int(zero.sum()), zero_reference_selected=int((zero & selected).sum()),
                        zero_reference_harm=float(harm[zero & selected].sum()),
                        missed_oracle_positive_rows=int((known & (reasons!=4) & (benefit>0)).sum()),
                        harmful_selected_rows=int((selected & (harm>0)).sum())))
    payload=dict(evaluation_sha256=run.digest(run.PUBLIC/'evaluation.json'), result_source='fresh_postfreeze_descriptive_diagnosis',
                 threshold_changes=False, new_training=False, independent_roles_read=False, views=rows)
    run.immutable_json(run.PUBLIC/'risk_diagnosis.json',payload)
    lines=['# Risk and Fallback Diagnosis','',
        'Post-freeze descriptive diagnostics, not a threshold search. Counts are dependent role/seed views.',
        'A predicted2% ratio is not a realized2% guarantee. Undefined zero-denominator ratios remain null.', '',
        '| Candidate | Known row-views | Selected | Harmful selected | Missed beneficial | Selected zero-reference |',
        '|---|---:|---:|---:|---:|---:|']
    for c in ('dimensionless','damped'):
        rr=[r for r in rows if r['candidate']==c]
        keys=('known_rows','selected_rows','harmful_selected_rows','missed_oracle_positive_rows','zero_reference_selected')
        lines.append('| '+c+' | '+' | '.join(str(sum(r[k] for r in rr)) for k in keys)+' |')
    lines += ['', '| Candidate | Reason | Row-views |','|---|---|---:|']
    for c in ('dimensionless','damped'):
        rr=[r for r in rows if r['candidate']==c]
        for reason in rows[0]['reasons']:
            lines.append(f"| {c} | {reason} | {sum(r['reasons'][reason] for r in rr)} |")
    lines += ['', '## Locality Risk Extremes', '',
        '| Candidate | View/site | Predicted selected harm ratio | Realized selected harm ratio |',
        '|---|---|---:|---:|']
    for c in ('dimensionless','damped'):
        rr=sorted([r for r in rows if r['candidate']==c and r['selected_actual_all_ratio'] is not None],
                  key=lambda r:r['selected_actual_all_ratio'],reverse=True)[:8]
        for r in rr:
            lines.append(f"| {c} | {r['group']}/{r['site']} | {r['selected_predicted_all_ratio']} | {r['selected_actual_all_ratio']} |")
    lines += ['', 'These ratios diagnose selected-set magnitude mismatch, not independent calibration.',
        'Do not rescue the model by selecting the best displayed seed/site or refitting thresholds on these outcomes.']
    (run.PUBLIC/'risk_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(views=len(rows), threshold_changes=False)))


if __name__=='__main__':main()
