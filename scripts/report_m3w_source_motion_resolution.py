"""Render complete fixed native-motion probe evidence without selecting a model."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/m3w-mpl-resolution')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/m3w-xdg-resolution')
from src.world_model.m3w_source_motion_resolution import registration, VARIANTS
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def main():
    plan=registration(ROOT); public=ROOT/plan['reports']
    prep=json.loads((public/'preparation.json').read_text())
    flow=json.loads((public/'extraction.json').read_text())
    probe=json.loads((public/'probes.json').read_text())
    verified=json.loads((public/'verification.json').read_text())
    rawpath=ROOT/plan['raw_label_archive']
    assert file_digest(rawpath)==probe['identity']['raw_labels_sha256']
    with np.load(rawpath,allow_pickle=False) as raw:
        positive=np.sum(raw['future_native']**2,axis=-1).max(1)>100
        support=dict(result_source='fresh_run_descriptive_counts_on_cached_verified_raw_labels',
            raw_archive_sha256=file_digest(rawpath),rows=int(positive.sum()),
            scoped_tracks=len(np.unique(raw['tracks'][positive])),
            recordings=len(np.unique(raw['records'][positive])),
            sites={str(s):dict(rows=int((positive & (raw['sites']==s)).sum()),
                scoped_tracks=len(np.unique(raw['tracks'][positive & (raw['sites']==s)])))
                for s in np.unique(raw['sites'])},overlapping_queries_are_not_independent=True)
    (public/'positive_support.json').write_text(json.dumps(support,indent=2)+'\n')
    lines=['# Native Motion: Resolution And Window Evidence','',
        'Source: fresh_run native decoding, motion extraction and fixed logistic fitting; '
        'cached_verified source population and old controls. No model selection or deployment.', '',
        '## Scope And Cost','',
        'All 15,430 stationary-history queries across 29 recordings and four explored SDD sites are retained. '
        'This is the approved offline supplied-annotation source diagnostic, not the full primary '
        'benchmark, independent confirmation or external transfer. Eight observed and twelve predicted '
        'steps use a stride of 12 raw frames. No seconds, metric, true-3D or foundation claim. '
        'Bookstore, main and outer populations remain unscored.', '',
        f"- Native past crops: {prep['crops']:,}; exact old-pixel reductions: {prep['exact_reductions']:,}.",
        f"- Native cache: {prep['bytes']/1024**2:.2f} MiB; decode/crop time: {prep['seconds']:.2f} s.",
        f"- Pair measurements: {4*flow['unique_pairs_per_variant']:,}; flow time: {sum(r['seconds'] for r in flow['records']):.2f} s.",
        f"- Logistic models: 64; fitting time: {sum(t['seconds'] for t in probe['trials']):.2f} s; max iterations: {max(max(t['iterations']) for t in probe['trials'])}.",
        '- Native arm64 CPU, four compute threads, single-process data loading; completed fits checkpointed separately.',
        '- No new neural trajectory training in this comparison. The four measurements do not establish body-motion truth.', '',
        '## Measurement Support','',
        '| Variant | Box support | Surround support | Mean box magnitude (annotation px/past pair) |',
        '| --- | ---: | ---: | ---: |']
    for v,a in flow['variants'].items():
        lines.append(f"| {v} | {100*a['box_support']:.3f}% | {100*a['surround_support']:.3f}% | {a['box_magnitude']['mean']:.6f} |")
    lines += ['', 'Support is a coverage/consistency proxy, not verified motion accuracy. Nominal '
        'window extent is matched; pixel lattice, polynomial support and pyramid operation are not '
        'perfectly isolated. All 23,890 lowpass_w45 pair features match the previous control exactly. '
        'Quality controls include flow-consistency and support flags; the motion contrast measures '
        'the added vectors and magnitudes, not all motion-derived information versus none.', '',
        '## Complete Probability Results','',
        'The supervision labels are exact raw future coordinates, used only for loss/evaluation. '
        'There are 728 strictly-greater-than-10-pixel cases. Earlier 739-label results are not substituted '
        'as controls; all new lowpass and native probes use the same exact labels. No post-outcome '
        'relabeling, threshold search or probe selection.', '',
        '| Target | Variant | Input | Equal-site Brier | Log loss | AUROC | AUPRC | Sites beating prevalence Brier |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for label in plan['targets']:
        for v in VARIANTS:
            for arm in plan['arms']:
                subset=[t for t in probe['trials'] if t['label']==label and t['variant']==v and t['arm']==arm]
                values=[np.mean([t['held'][m] for t in subset]) for m in ('brier','log_loss','auroc','auprc')]
                wins=sum(t['held']['brier']<t['prior']['brier'] for t in subset)
                lines.append('| '+ ' | '.join([label,v,arm]+[f'{x:.8f}' for x in values]+[f'{wins}/4'])+' |')
    lines += ['', '## Paired Contrasts','',
        'Positive values favor the first/new representation; Brier/log-loss values are reductions. '
        'All contrasts are reported, not a selected winner. There are 2,000 paired site resamples, four '
        'already-explored sites with shared training folds. These are unadjusted exploratory '
        'intervals, not independent confirmation, multiplicity-controlled discovery or risk calibration.', '',
        '| Target | Contrast | Metric | Difference | Conditional 95% CI |', '| --- | --- | --- | ---: | --- |']
    for c in probe['contrasts']:
        lo,hi=c['conditional_four_site_ci95']
        lines.append(f"| {c['label']} | {c['name']} | {c['metric']} | {c['equal_site_difference']:.8f} | [{lo:.8f}, {hi:.8f}] |")
    lines += ['', '## Per-Site Readout','',
        '| Site | Target | Variant | Input | Brier | Prevalence Brier | AUROC | Positives/rows |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for t in probe['trials']:
        h=t['held']; lines.append(f"| {t['site']} | {t['label']} | {t['variant']} | {t['arm']} | {h['brier']:.8f} | {t['prior']['brier']:.8f} | {h['auroc']:.6f} | {h['positives']}/{h['rows']} |")
    lines += ['', '## Verification And Boundaries','',
        f"{verified['exact_flow_replays']:,} exact pair replays; 64 exact coefficient replays; 128 future-label "
        f"poison queries; 48 prohibited training-role rejections. {verified['immutable_artifacts']} artifacts "
        'unchanged through completed decoder/extractor/probe resume with zero new work.', '',
        'The corpus constructor loads broader label arrays for provenance, but the image/flow '
        'functions take no labels and the input poison tests leave features unchanged. '
        'Offline interpolated supplied annotations are not certified sensor-as-of observations.', '',
        'No trajectory ADE/FDE, neural selector advantage or safe intervention is established by '
        'probability ranking alone. Four explored sites and rare motion remain limiting. '
        'Independent calibration/confirmation, main predictor comparisons and final paper evidence '
        'are still incomplete. No Stage5C or SMC.', '',
        f"Probes SHA256: `{file_digest(public/'probes.json')}`.",
        f"Verification SHA256: `{file_digest(public/'verification.json')}`.", '']
    # This is a generated report, never an input to model selection.
    (public/'complete_results.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for row,label in enumerate(plan['targets']):
        for col,metric in enumerate(('brier','auroc')):
            ax=axes[row,col]
            for i,v in enumerate(VARIANTS):
                c=next(c for c in probe['contrasts'] if c['label']==label and c['metric']==metric
                       and c['name']==v+'_motion_vs_quality')
                lo,hi=c['conditional_four_site_ci95']; mean=c['equal_site_difference']
                ax.errorbar(mean,i,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color='#206b53')
            ax.axvline(0,color='#999999',linewidth=1)
            ax.set_yticks(range(4),list(VARIANTS))
            ax.set_title('Any future annotation change' if row==0 else 'Maximum excursion >10 annotation px')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.set_xlabel('Brier reduction' if metric=='brier' else 'AUROC increase')
    fig.suptitle('Added observed motion vs matched quality control\nFour explored source sites; conditional intervals, not confirmation')
    svg=public/'motion_information.svg'; fig.savefig(svg)
    fig.savefig('/tmp/m3w-source-motion-resolution.png',dpi=140); plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(report=str(public/'complete_results.md'),figure=str(svg))))


if __name__=='__main__': main()
