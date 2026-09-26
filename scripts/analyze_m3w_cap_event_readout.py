"""Post-readout descriptive diagnosis; never selects a model or threshold."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_exceedance as run
import numpy as np


def main():
    source = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    flat = []
    for row in source:
        for arm in ['linear', 'mlp']:
            m = row['metrics'][arm]; s = row['support'][0]
            receipt = json.loads((run.PRIVATE/'heads'/row['tag']/arm/'complete.json').read_text())
            flat.append(dict(tag=row['tag'], pair=row['pair'], arm=arm, seed=row['seed'],
                outer=row['outer'], positive=s['positive'], known=s['known'],
                positive_agents=s['positive_agents'], positive_recordings=s['positive_recordings'],
                fitting_prior=receipt['fitting_prior'], held_prevalence=m['prevalence'],
                predicted_rate=m['predicted_rate'], AUROC=m['AUROC'], AP=m['AP'],
                BCE=m['BCE'], prior_BCE=m['prior_BCE'], Brier=m['Brier'], prior_Brier=m['prior_Brier'],
                top10_overshoot_mass=m['top10_overshoot_mass']))
    with (run.PUBLIC/'view_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    summaries = {}
    for pair in ['full', 'motion_only']:
        for arm in ['linear', 'mlp']:
            rows = [r for r in flat if (r['pair'], r['arm']) == (pair, arm)]
            au = [r['AUROC'] for r in rows if r['AUROC'] is not None]
            summaries[pair+'__'+arm] = dict(views=len(rows),
                BCE_better_than_fitting_prior=sum(r['BCE'] < r['prior_BCE'] for r in rows),
                Brier_better_than_fitting_prior=sum(r['Brier'] < r['prior_Brier'] for r in rows),
                unavailable_ranking_views=sum(r['AP'] is None for r in rows),
                median_AUROC_supported_views=float(np.median(au)),
                minimum_positive_agents=min(r['positive_agents'] for r in rows),
                minimum_positive_recordings=min(r['positive_recordings'] for r in rows),
                median_absolute_probability_mean_gap=float(np.median([abs(r['predicted_rate']-r['held_prevalence']) for r in rows])),
                median_absolute_fitting_prior_to_held_rate_gap=float(np.median([abs(r['fitting_prior']-r['held_prevalence']) for r in rows])))
    aggregate = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    failed_components = [r for r in aggregate['contrasts'] if r['arm'] == 'mlp' and r['pair'] == 'full'
        and r['contrast'] in ['BCE_gain_prior', 'AP_gain_envelope', 'capture_gain_envelope'] and r.get('sign') != 'positive']
    run.immutable_json(run.PUBLIC/'post_readout_diagnosis.json', dict(summary=summaries,
        primary_nonpassing_components=failed_components, result_source='fresh_run_post_readout_descriptive',
        independent_views=False, causal_failure_attribution_proven=False, thresholds_changed=False,
        model_selected=False, source_readout=run.artifact(run.PUBLIC/'readout.json')))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': main()
