"""Rebuild the decision supplement and source-influence audit from verified aggregates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_decision_manuscript_v12 import DATA, HOME, PUBLIC, read, sha, source
from src.evaluation.m3w_joint_path_audit import bootstrap

REPLAY = DATA/'review_replay_v1/review_replay.zip'
REPLAY_SHA = '577bd644da33ad397903529f3cbb7cb1378716a047883284c37a4b68396f43fe'
STYLE = DATA/'evidence_manuscript_v10/paper_build/cvpr.sty'
DEST = DATA/'evidence_manuscript_v12/supplement'
COMPARISONS = {
    'joint_vs_independent': 'Joint vs. independent',
    'joint_vs_scene': 'Joint vs. shared family',
    'joint_vs_pair_additive': 'Joint vs. additive',
    'joint_vs_cv': 'Joint vs. CV',
    'joint_vs_ols4': 'Joint vs. OLS4',
}


def source_influence(cells):
    """Average seeds before leaving one source out; never omit missing cells."""
    x = np.asarray(cells, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or len(x) < 2:
        raise ValueError('A source by seed grid with multiple sources is required')
    if not np.isfinite(x).all():
        return None
    by_source = x.mean(axis=1)
    loo = [float(np.delete(by_source, k).mean()) for k in range(len(x))]
    return dict(mean=float(by_source.mean()), positive_sources=int((by_source > 0).sum()),
                zero_sources=int((by_source == 0).sum()), sources=len(x),
                leave_one_source_out_range=[min(loo), max(loo)],
                seed_mean_range=[float(x.mean(0).min()), float(x.mean(0).max())])


def contrast_influence(packet, canonical):
    groups = sorted(set(packet['roster'].values()))
    seeds = packet['seeds']
    data = packet['joint_cells']
    if packet['schema_version'] != 1 or (len(packet['roster']), len(groups), len(seeds)) != (34, 14, 3):
        raise ValueError('Unexpected development role')
    if (len(set(seeds)) != len(seeds) or sorted(data) != groups
            or any(set(data[g]) != set(map(str, seeds)) for g in groups)):
        raise ValueError('Expected every development source/seed cell')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    result = {}
    for key in COMPARISONS:
        result[key] = {}
        for metric in ('observed_ade_reduction', 'grid_ade_lower', 'grid_ade_upper'):
            values = [[data[g][str(s)][key][metric] for s in seeds] for g in groups]
            stat = bootstrap(values, draw)
            if stat != canonical[key][metric]:
                raise ValueError(f'Paired replay does not match: {key}/{metric}')
            result[key][metric] = source_influence(values)
    return result


def macros(risk, support, influence):
    table = []
    for label, key in (
            ('CV', 'constant_velocity_causal_fd'), ('OLS4', 'history_ols4_velocity'),
            ('Independent', 'independent'), ('Shared family', 'scene'),
            ('Additive', 'pair_additive'), ('Joint', 'joint'), ('Screened joint', 'gated_joint')):
        row = risk[key]
        table.append(f'{label} & {row["switches"]:,} & {row["supported_views"]} & '
                     f'{row["undefined_easy_views"]} & {row["all_risk_over_2_views"]} & '
                     f'{row["worst_recording_p95_known_ade_pixel"]:.3f} '+r'\\')
    source_rows = []
    for key, label in COMPARISONS.items():
        obs = influence[key]['observed_ade_reduction']
        lower = influence[key]['grid_ade_lower']
        if obs is None or lower is None:
            raise ValueError('Current table requires all source/seed cells')
        cols = [label]
        for a in (obs, lower):
            lo, hi = a['leave_one_source_out_range']
            cols.extend([str(a['positive_sources']), f'$[{lo:+.4f}, {hi:+.4f}]$'])
        source_rows.append(' & '.join(cols)+' '+r'\\')
    values = dict(SuppRiskRows='\n'.join(table), SuppSourceRows='\n'.join(source_rows),
                  PairObservedSteps=f'{support["observed_pair_steps"]:,}',
                  PairRequestedSteps=f'{support["requested_pair_steps"]:,}',
                  PairObservedEndpoints=f'{support["observed_terminal_pairs"]:,}',
                  PairRequestedEndpoints=f'{support["requested_terminal_pairs"]:,}')
    return '\n'.join("\\newcommand{\\"+k+"}{"+v+"}" for k, v in values.items())+'\n'


def build(out, archive, style):
    bindings = {}
    analyses = {}
    for k in ('dronecrowd_transport_candidate_v1', 'moving_joint_path_audit_v1'):
        rec = read(HOME/k/'replay_execution.json')
        digest = rec['analysis_sha256'] if 'analysis_sha256' in rec else rec['artifact']['sha256']
        analyses[k] = read(source(k+'/analysis.json', digest, bindings))
    transport, joint = (analyses[k] for k in ('dronecrowd_transport_candidate_v1', 'moving_joint_path_audit_v1'))
    if sha(archive) != REPLAY_SHA:
        raise ValueError('Expected the verified development aggregate archive')
    if sha(style) != read(HOME/'evidence_manuscript_v11/build_verification.json')['cvpr_style_sha256']:
        raise ValueError('Expected the verified CVPR 2027 style')
    with ZipFile(archive) as zipped:
        raw = zipped.read('statistics.json')
        if hashlib.sha256(raw).hexdigest() != json.loads(zipped.read('manifest.json'))['statistics.json']:
            raise ValueError('Aggregate archive content mismatch')
    result = contrast_influence(json.loads(raw), joint['comparisons'])
    out.mkdir(parents=True, exist_ok=True)
    for a, b in ((PUBLIC/'supplement.tex', out/'supplement.tex'), (style, out/'cvpr.sty')):
        shutil.copyfile(a, b)
    (out/'supplement_values.tex').write_text(macros(transport['grouped']['risk'], joint['support'], result))
    audit = dict(result_source='fresh_source_influence_from_verified_exposed_development_aggregates',
                 independent_confirmation=False, external_scores_read=False, deployment_changed=False,
                 stage5c_executed=False, smc_enabled=False, development_archive_sha256=REPLAY_SHA,
                 source_bindings=bindings,
                 source_influence=result)
    (out/'source_influence.json').write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=REPLAY)
    parser.add_argument('--style', type=Path, default=STYLE)
    parser.add_argument('--output-dir', type=Path, default=DEST)
    args = parser.parse_args()
    build(args.output_dir, args.archive, args.style)


if __name__ == '__main__':
    main()
