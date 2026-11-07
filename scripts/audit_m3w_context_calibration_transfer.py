"""Verify matching coverage and preservation transfer on frozen development outputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_context_calibration as study

run,np = study.run,study.np


def main():
    path = study.PUBLIC/'analysis.json'
    a = run.read(path)
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Exact frozen evaluation required')
    detail = [v for c in run.read(study.PUBLIC/'freeze.json')['cells'] for v in c['details']]
    records = a['records']
    if len(detail)!=102 or len(records)!=102:
        raise ValueError('Full view roster required')
    groups = sorted({r['source'] for r in records})
    rows = [[r for r in records if r['source']==s and r['seed']==k]
            for s in groups for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(groups),size=(3000,len(groups)))
    def hard(arm):
        v = []
        for g in rows:
            pair = [(r['metrics']['base']['subsets']['hard'],r['metrics'][arm]['subsets']['hard'])
                    for r in g if r['metrics']['base']['subsets']['hard']['known_rows']]
            mass = sum(p['known_rows']*p['floor_ADE_pixel'] for p,_ in pair)
            if not mass:
                return None
            v.append(100*sum(p['known_rows']*(p['model_ADE_pixel']-q['model_ADE_pixel']) for p,q in pair)/mass)
        v = np.asarray(v).reshape(len(groups),3).mean(1)
        return dict(point=float(v.mean()),interval95=np.quantile(v[draw].mean(1),[.025,.975]).tolist())
    info = dict(source=run.artifact(path),views=102,
        near_blind_different_peers=sum(r['near']['peer_records']!=r['blind']['peer_records'] for r in detail),
        near_blind_different_threshold=sum(r['near']['threshold']!=r['blind']['threshold'] for r in detail),
        near_far_different_threshold=sum(r['near']['threshold']!=r['far']['threshold'] for r in detail),
        fail_preservation={a:sum(not study.base.guard.preserves(r['metrics']['base']['risk'],
            r['metrics'][a]['risk']) for r in records) for a in study.ARMS[1:]},
        hard_known_gain_vs_base_pp={a:hard(a) for a in study.ARMS[1:]},
        no_thresholds_selected_or_modified=True,independent_outcomes_opened=False)
    run.once(study.PUBLIC/'transfer_diagnostic.json',info)
    print(info)


if __name__=='__main__':
    main()
