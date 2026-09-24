"""Render all zero-reference controls without selecting a deployment winner."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/zero_atom_v1'


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    lines=['# Explicit Zero-Reference Atom: All Controls','',
        'Four development-exposed SDD sites, three seeds, obs8/pred12 stride12 annotation pixels.',
        'Equal-physical-site gains over CV; not historical raw t50, strongest-baseline or independent-confirmation claims.','',
        '| Action | Policy | ADE gain % | FDE gain % | Hard gain % | Worst positive-easy degradation % | Zero-CV harms, window/seed | Mean switches |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for action in ('damped_velocity_005','transformer','eqmotion'):
        for p in a['policies']:
            r=a['summary'][action+'__'+p]
            easy=max(-x['gain_percent'] for s in r['seeds'].values() for x in s['subsets']['positive_easy']['by_scene'].values())
            harms=sum(s['zero_CV_harmed'] for s in r['seeds'].values())
            count=np.mean([s['selected'] for s in r['seeds'].values()])
            lines.append(f"| {action} | {p} | {r['ADE']['equal_scene_gain_percent']:.6f} | {r['FDE']['equal_scene_gain_percent']:.6f} | {r['subsets']['hard']['equal_scene_gain_percent']:.6f} | {easy:.6f} | {harms} | {count:.1f} |")
    lines+=['','All three guard/control counts match within every recording/frame/seed, not just globally.',
        'Numerically failed matched proposals retain their feasible incumbent and are not called optimal.',
        'Probability zero means no weighted positive event in any visited source leaf, not calibrated impossibility.',
        'Unknown and incomplete futures remain indexed; full-grid bounds are in analysis.json.','',
        '## All Registered Contrasts','','3000 paired resamples of four physical sites, nominal conditional development intervals.','']
    for k,v in a['contrasts'].items(): lines+=['### '+k,'','```json',json.dumps(v,indent=2),'```','']
    lines+=['## Solver','','```json',json.dumps(a['solver'],indent=2),'```','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    lines=['# Fresh Leaf Readout Fits and Support','',
        '36 new source-only leaf-frequency fits on 36 cached_verified forest partitions. No new forest or neural forecast training.',
        'Weighted fitting Brier is an in-source fitting diagnostic, not independent calibration.','',
        '| View | Action | Effective unique rows | Zero-reference rows | Moving zero-reference rows | Zero-reference weighted draws | Fitting Brier | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in a['fits']:
        s=r['support']; lines.append(f"| {r['view']} | {r['action']} | {s['effective_unique']} | {s['zero_effective']} | {s['zero_effective_moving']} | {s['zero_draws']} | {s['training_weighted_brier']:.7f} | {r['seconds']:.3f} |")
    lines+=['','## Held-Source Event Diagnostic','','All held sites remain design-exposed. No probability threshold was selected from these numbers.','',
        '| View | Action | Complete labels | Zero-CV rows | Moving zero-CV | Moving zero-CV admitted | Complete-label Brier |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in a['held_event_quality']:
        lines.append(f"| {r['view']} | {r['action']} | {r['known']} | {r['zero']} | {r['moving_zero']} | {r['moving_zero_admitted']} | {r['brier_complete']:.7f} |")
    (PUBLIC/'fit_support.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(fits=len(a['fits']),rows=len(a['summary']))))


if __name__=='__main__': main()
