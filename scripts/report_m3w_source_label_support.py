"""Public-scalar report for the complete causal SOURCE support probe."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_source_label_support_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_source_label_support_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read=lambda p:json.loads(p.read_text())
    s=read(PUBLIC/'summary.json');v=read(PUBLIC/'verification.json');iv=read(PUBLIC/'inference_verification.json')
    if (not v['all144_groups_exact'] or not iv['all18_probability_arrays_exact']
            or v['summary_sha256']!=sha(PUBLIC/'summary.json')):
        raise ValueError('Complete prediction and scalar replay required')
    train=read(PUBLIC/'training_freeze.json');pilot=read(PUBLIC/'pilot.json')
    fit=[read(ROOT/r['path']) for r in train['fits']]
    for r,ref in zip(fit,train['fits']):
        if sha(ROOT/ref['path'])!=ref['sha256'] or r['trees']!=128:raise ValueError('Complete model receipts required')
    groups=[read(p) for p in sorted((PUBLIC/'groups').glob('*.json'))]
    if len(groups)!=144:raise ValueError('All original groups required')
    fmt=lambda r:'not estimable' if r['point'] is None else f'{r["point"]:.4f} [{r["CI95"][0]:.4f}, {r["CI95"][1]:.4f}]'
    lines=['# Causal Future-Label-Support Probe', '', '## Material Passport', '',
        'Fresh_run18 full ExtraTrees fits, a complete first-model refit, real',
        'recovery pilot and full prediction/metric replay. Cached_verified SOURCE',
        'arrays/geometry/detector-history/role definitions. This report itself reads',
        'public scalar evidence, not raw data or fresh model inference.', '',
        '## Complete Held-SOURCE Results', '',
        '|Arm|AUROC [95% CI]|AUPRC [95% CI]|Unknown prevalence|AUPRC minus prevalence [95% CI]|Brier|ECE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    num=lambda r:'N/A' if r['point'] is None else f'{r["point"]:.4f}'
    for arm in ('geometry','geometry_detector'):
        r=s['arms'][arm]
        lines.append(f'|{arm}|{fmt(r["AUROC"])}|{fmt(r["AUPRC"])}|{num(r["positive_rate"])}|{fmt(r["AUPRC_minus_prevalence"])}|{num(r["Brier"])}|{num(r["ECE"])}|')
    lines+=['',f'Registered detector-minus-geometry AUROC effect: {fmt(s["primary_effect"])}.',
        'The primary is retained even if geometry alone has a better result.',
        'The unknown-prevalence value is an evaluation reference, not a held-label',
        'probability fed into inference. No probability calibration or threshold',
        'selection is fitted on held outcomes. All point estimates are equal-locality',
        'means, including prevalence; they are not pooled row-level event rates.',
        'Scores need not be well calibrated.', '',
        '## What This Tests', '',
        'The positive event is that none of twelve requested future labels exists',
        'in the release, not physical occlusion. Past geometry476 and detector64',
        'are causal cached-verified fields. The future mask supplies supervision',
        'and evaluation labels only; no mask, endpoint, eventual track length,',
        'central velocity or test statistic enters the feature matrix.',
        'Each forest fits its original four SOURCE localities and predicts the',
        'excluded eight. There are149,510windows,74recordings,twelve localities.',
        'The nine seed/fold views per arm repeat data, not independent people.',
        'Intervals use3,000locality resamples after averaging two excluded folds',
        'and actual forest seeds17/29/43. SOURCE has informed research choices;',
        'this is crossfit development evidence, not independent confirmation.', '',
        '## Fitting And Domain Breakdown', '',
        '|Arm|Fitting AUROC min/max across all nine models|Held-SOURCE AUROC min/max across localities|',
        '|---|---:|---:|']
    for arm in ('geometry','geometry_detector'):
        vals=[r['fitting_scores']['AUROC'] for r in fit if r['arm']==arm]
        held=list(s['arms'][arm]['AUROC']['by_locality'].values())
        a=f'{min(vals):.4f}/{max(vals):.4f}' if all(x is not None for x in vals) else 'N/A'
        b=f'{min(held):.4f}/{max(held):.4f}' if all(x is not None for x in held) else 'N/A'
        lines.append(f'|{arm}|{a}|{b}|')
    lines+=['','Fitting scores are resubstitution on other sites, not paired held validation;',
        'their apparent gap does not isolate one cause of domain shift.', '',
        '|SOURCE locality|Geometry AUROC|Geometry+detector AUROC|Geometry+detector AUPRC|',
        '|---|---:|---:|---:|']
    for site in s['arms']['geometry']['AUROC']['by_locality']:
        fields=[s['arms'][arm][field]['by_locality'][site]
                for arm,field in (('geometry','AUROC'),('geometry_detector','AUROC'),('geometry_detector','AUPRC'))]
        lines.append('|'+site+'|'+'|'.join('N/A' if x is None else f'{x:.4f}' for x in fields)+'|')
    lines+=['','## Recording51 At Locality082', '',
        '|Arm|Trial|Unknown rows / total|Unknown score mean|Known score mean|AUROC|AUPRC|',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['site']=='eu-locality-082' and '51' in g['recordings']:
            r=g['recordings']['51']
            f=lambda x:'N/A' if x is None else f'{x:.4f}'
            lines.append(f'|{g["arm"]}|{g["trial"]}|{r["unknown_rows"]}/{r["rows"]}|{f(r["unknown_score_mean"])}|{f(r["known_score_mean"])}|{f(r["AUROC"])}|{f(r["AUPRC"])}|')
    lines+=['','No scene/recording is dropped, and one-class recording groups remain',
        'not estimable rather than favorable perfect scores. A useful ranking here',
        'would still not prove safe intervention or eliminate unknown-row bounds.', '',
        '## Compute And Reproduction', '',
        f'Primary forest trees:{train["primary_trees"]};first full refit:{train["first_full_refit_trees"]};pilot:{pilot["actual_trees"]}.',
        f'Training PID{train["pid"]}:{train["seconds"]:.3f}s,peak{train["peak_RSS_bytes"]:,}bytes.',
        f'Complete prediction replay PID{iv["pid"]}:{iv["seconds"]:.3f}s.',
        'All18probability arrays and144model/locality metric groups reproduce exactly.',
        'Forest training uses four native threads, deterministic single-thread',
        'probability reduction,per32-tree checkpoints and unchanged resume.',
        'These are actual fitted forests, not new Torch dynamics or NumPy fallback.', '',
        '```sh', '.venv-pytorch/bin/python scripts/run_m3w_source_label_support.py train --resume',
        '.venv-pytorch/bin/python scripts/report_m3w_source_label_support.py', '```', '',
        'Resume applies to unfinished work; complete stage receipts are immutable.',
        'The original completed replay and public-scalar reconstruction are distinct.', '',
        '## Limits And Next Action', '',
        'This probe does not train a selector, use its probabilities for admission,',
        'or pass a new2%recording-risk contract. The primary feature-ablation effect',
        'must be separated from absolute label-support predictability. It cannot',
        'establish neural dynamics, scene/goal contribution or causal physical visibility.',
        'If signal is useful, the next experiment must compare explicit unknown-row',
        'disagreement-bound supervision against ordinary known-only cost learning,',
        'with calibration and final confirmation still closed. If it fails, diagnose',
        'the past-input/label-process gap rather than giving future availability as input.',
        'Pixels/rawstride12/released silver only; not metric,seconds,human gold,',
        'physical safety,true3D or foundation. No deployment upgrade,Stage5C or SMC.', '']
    (PUBLIC/'readout.md').write_text('\n'.join(lines))
    receipt=dict(result_source='verified_public_scalar_reconstruction_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        inference_verification_sha256=sha(PUBLIC/'inference_verification.json'),
        reporter_sha256=sha(Path(__file__)),independent_confirmation=False,deployment_changed=False)
    (PUBLIC/'report_sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
