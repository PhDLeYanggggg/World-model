"""Replay frozen development choices and attribute the original recording risk failures."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_structured_imitation as student
from scripts.build_m3w_decision_supplement_v12 import source_influence
from src.evaluation.m3w_recording_forecast_comparison import merge_comparisons
from src.evaluation.m3w_recording_risk_attribution import summarize

run, prev, np = student.run, student.prev, student.np
ARMS = ('gain', 'support_first', 'shortfall', 'student_nomination', 'student_full', 'student_screen')
PAIRS = {f'{a}_vs_gain': ('gain', a, student.BASE) for a in ('student_nomination', 'student_full')}
OUTPUT = student.PUBLIC.parent/'evidence_manuscript_v14/risk_attribution.json'


def analyze():
    cut = student.register()
    frozen = prev.receipt(student.PUBLIC, 'freeze')
    if frozen['score_labels_read'] or frozen['registration'] != run.sha(student.REG):
        raise ValueError('Expected predictions fixed before score')
    analysis_path = student.PUBLIC/'analysis.json'
    original = prev.receipt(student.PUBLIC, 'replay')['grouped']
    if original != run.read(analysis_path)['grouped']:
        raise ValueError('Development analysis does not match replay receipt')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    blocks = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in blocks.items()}
    rows = []
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = dict(zip(student.ARMS, z['choices'], strict=True))
        for rec, (ix, queries) in blocks.items():
            paths = {student.BASE: cv[ix], **{
                a: run.model.selected_paths(choices[a][ix], bank[ix], cv[ix]) for a in ARMS
            }}
            changed = np.stack([(paths[a] != cv[ix]).any(axis=(1, 2)) for a in student.MATCHED])
            for at in queries:
                n = changed[:, at].sum(1)
                if not np.all(n == n[0]):
                    raise ValueError('Frozen genuine switch counts do not match')
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix],
                                                      rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
        print(f'Seed {seed}: {len(rows)} recording-seed views audited', flush=True)
    replay = prev.old.prior.grouped_readout(rows, roster, student.SEEDS)
    for key in ('risk', 'effects'):
        if any(v != original[key][a] for a, v in replay[key].items()):
            raise ValueError(f'Frozen original {key} did not replay')
    risk = {a: summarize([r['metrics'][a]['risk'] for r in rows]) for a in ARMS}
    groups = sorted(set(roster.values()))
    merged = {}
    for pair in PAIRS:
        cells = [[merge_comparisons([r['comparisons'][pair] for r in rows
                                    if (r['source'], r['seed']) == (g, s)])
                  for s in student.SEEDS] for g in groups]
        merged[pair] = {key: dict(replay['effects'][pair][key],
                                  influence=source_influence([[c[key] for c in g] for g in cells]))
                        for key in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
    return dict(analysis_sha256=run.sha(analysis_path), groups=len(groups), recordings=len(roster),
                views=len(rows), seeds=list(student.SEEDS), risk=risk, utility=merged,
                scorer_replay_exact=True, inference_or_gates_changed=False,
                result_source='fresh_attribution_from_hash_verified_exposed_development_predictions',
                external_access=False, independent_confirmation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run.once(args.output, run.stable(analyze()))
