"""Extend verified revision6 with the fixed shared-reference and scene ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import build_m3w_evidence_manuscript_v6 as prior
from scripts import report_m3w_feature_cost_evidence as feature

DEST = prior.PUBLIC/'evidence_manuscript_v7'


def insert_before(text, marker, extra):
    if text.count(marker) != 1:
        raise ValueError('One stable section boundary required')
    return text.replace(marker, extra+'\n\n'+marker, 1)


def manuscript(evidence, original):
    shared, scene = feature.STUDIES[-2:]
    held = {(r['study'], r['policy']):r for r in evidence['policies'] if r['scope']=='inner_held'}
    original = original.replace('a single-factor trial.', 'a single-factor trial within that original comparison.')
    abstract = ('We also test parameter-matched reference sharing and a causal static-image-region '
        'feature ablation. Scene context improves known-label routing, but neither new ablation '
        'establishes a safe neural increment. ')
    original = insert_before(original, '## 1. Introduction', abstract)
    method = '''### 5.5 Candidate-Invariant Reference Sharing

The floor reference and its easy-event mass belong to a query, not to the proposed
action. A frozen 36-state fitting diagnosis finds substantial variation between
candidate estimates of these invariant targets. We evaluate a shared-reference
head that reuses the existing network with a single causal floor token and zero
action code. Candidate tokens still supply benefit, harm and easy harm. The model
retains 25,157 parameters and the original initialization, normalizers, draws, loss
and update budget. Exact reference invariance is an architectural constraint,
not a theorem about cost accuracy or empirical risk.

### 5.6 Static Image-Region Context

Relative-motion features discard global image translation. Static region context
may therefore distinguish situations with similar local motion but different
image positions. The publisher releases image-coordinate place polygons [6].
We bind 14 actual cameras to 74 fitting recordings by raw camera ID, rather than by
conservative locality grouping. Some locality groups contain two cameras.
The audit verifies no gross expanded-bbox mismatch; it does not establish metric
calibration, walkability, obstacles, an exit label or exact viewport boundaries.
Missingness is not monotonic in polygon distance, so edge-driven censoring is
not established as the explanation for policy failure.

Eighteen fields encode normalized current/past-start position, current/mean
velocity, past signed region distance and changes, and distances/inside fractions
of a proposed causal rollout. Future positions and label availability never
enter this feature function. A zero-initialized 18x64 projection adds 1,152 weights
to the existing hidden layer, for 26,309 total parameters. Normalization uses only
physical fitting candidates. The comparison does not stack reference sharing
with the scene change. A full 2,000-update disabled branch exactly reproduces
old network weights, Adam moments, sampled rows and loss traces.

Both ablations retain six contexts, three controller seeds, three banks and all
54 planned fits. They add 216,000 actual primary updates to the four-study record,
for 324 controller heads and 486,000 updates total. The trajectory producer stays
fixed at seed 43. Every new endpoint is committed before first readout, and each
of the 144 group readouts is reproduced in a separate process.'''
    original = insert_before(original, '## 6. Experimental Protocol', method)
    rows = ['### 7.1 Reference and Static-Context Ablations', '',
        '| Additional controller | Physical-only gain % | Mixed gain % | Mixed risk-supported views |',
        '|---|---:|---:|---:|']
    for study, label in ((shared, 'Shared reference'), (scene, 'Static image-region context')):
        physical, mixed = held[study, 'physical_only'], held[study, 'physical_neural']
        rows.append(f"| {label} | {physical['known_ADE_gain_floor_percent']:.6f} | "
            f"{mixed['known_ADE_gain_floor_percent']:.6f} | {mixed['supported_record_head_views']}/186 |")
    contrast = evidence['inner_held_mixed_minus_physical'][scene]['full_known_pp']
    harm = evidence['calibration_MSE_new_minus_original'][scene]['inner_held']['physical_neural']['harm']
    rows += ['', 'Reference sharing does not establish an incremental gain over either its',
        'unshared parent or the physical-only bank. Static context raises mixed known-',
        'label utility relative to the no-scene parent by 0.428982 percentage points,',
        'conditional 95% locality interval [0.053706,1.044617]. The matched-count',
        'utility difference is 0.112189 points [0.028429,0.232094]. However, the full',
        'completion-lower change is -0.405680 points [-1.185725,0.126229]. The known-',
        'label component effect is not robust positive utility under missing futures.', '',
        f"For static context, mixed minus physical-only is {contrast['point']:.6f} points,",
        f"interval [{contrast['CI95'][0]:.6f},{contrast['CI95'][1]:.6f}]. Neural availability does not",
        'improve this decision study. The mixed policy selects 32,219 row views, including',
        '518 unknown outcomes, and supports only 23/186 required recording/head views.',
        'Its 20 undefined easy-risk views are not passed safety checks.', '',
        f"The normalized harm-moment MSE change is {harm['point']:.6f},",
        f"conditional interval [{harm['CI95'][0]:.6f},{harm['CI95'][1]:.6f}]. This is a squared-error",
        'difference,not a percentage or pixel-error change. Lower overall harm-MSE',
        'coexists with failed selected-risk support. Candidate selection may concentrate',
        'underestimated risks,but that mechanism remains a hypothesis to test directly.',
        'The five-moment comparisons are exploratory,not a multiplicity-controlled',
        'familywise claim. All intervals remain conditional on exposed SOURCE data.', '',
        '![Fixed-bank controller evidence](../feature_cost_figures_20261029/neural_increment_and_record_risk.png)', '',
        'Figure 1. Paired neural increments and required risk support across the six',
        'fixed studies. Repeated contexts/head seeds do not create independent recordings.']
    original = insert_before(original, '## 8. Failure Analysis and Limitations', '\n'.join(rows))
    original = insert_before(original, '## 9. Reproducibility and Responsible Reporting', '''
The new scene result narrows rather than resolves the failure. Static layout
proxies can improve cost prediction, but the available neural forecast still
does not add reliable value over physical alternatives. Neither average moment
MSE nor candidate-invariant denominators certifies the population selected by
the policy. A prospective frozen-TRAIN audit of selected versus all-eligible
moment errors is the next distinguishing test, not another favorable threshold.
No independent outcomes are opened to choose that repair.'''.strip())
    return original+'\n6. Wolff, N., and Perry, L. Pedestrian Trajectory Dataset of Public European Squares.\nVersion 1.1, 2026. https://zenodo.org/records/18267205\n'


def build():
    old = prior.manuscript(prior.build())
    if old != (prior.DEST/'manuscript.md').read_text():
        raise ValueError('Original revision6 changed')
    evidence = feature.build()
    if evidence != json.loads(feature.DEST.with_suffix('.json').read_text()):
        raise ValueError('Verified feature evidence export changed')
    return manuscript(evidence, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); text = build(); DEST.mkdir(exist_ok=True)
    manifest = dict(result_source='cached_verified_manuscript_export_not_fresh_training_or_statistics',
        original_revision6_sha256=prior.sha(prior.DEST/'manuscript.md'),
        feature_evidence_sha256=prior.sha(feature.DEST.with_suffix('.json')),
        manuscript_sha256=hashlib.sha256(text.encode()).hexdigest(), words=len(text.split()),
        anonymous_author_details_included=False, independent_confirmation=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    for name, value in {'manuscript.md':text, 'evidence_manifest.json':json.dumps(manifest, indent=2)+'\n'}.items():
        path = DEST/name
        if args.check or path.exists():
            if path.read_text() != value:
                raise ValueError('New manuscript revision required')
        else:
            path.write_text(value)
    print(json.dumps(dict(exact_check=args.check, words=len(text.split()), submission_ready=False)))


if __name__ == '__main__':
    main()
