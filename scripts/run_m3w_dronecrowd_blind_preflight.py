"""Stream the full past-only roster under both unassigned DroneCrowd proposals."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
import platform
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_cross_archive import DATA, HOME, digest, once, read
from src.data_unification.m3w_dronecrowd_cache import DroneCrowdSceneWindows
from src.data_unification.m3w_dronecrowd_frozen_inputs import adapt_past
from src.evaluation.m3w_dronecrowd_windows import PROBES
import numpy as np

PUBLIC = HOME/'dronecrowd_blind_preflight_v1'
PROPOSAL = HOME/'dronecrowd_external_role_proposal_v1/alternatives.json'
CONSTRAINT = HOME/'dronecrowd_grouping_v2/exclusion_constraints.json'
SCREEN = HOME/'dronecrowd_cross_archive_v1/analysis.json'
CACHE = HOME/'dronecrowd_recordings_v1/analysis.json'
FORECASTER_COVERAGE = HOME/'dronecrowd_combined_forecaster_coverage_v1/analysis.json'
CONTROLLER_COVERAGE = HOME/'dronecrowd_controller_coverage_v1/analysis.json'
PARTITIONS = ('split_calibration_confirmation', 'frozen_zero_shot')


def role_sets(options, recording_group):
    """Validate proposals without converting either one into assigned roles."""
    universe = set(recording_group)
    excluded = set(options['excluded']['recordings'])

    def check(role):
        clips, groups = set(role['recordings']), set(role['groups'])
        if (len(clips) != len(role['recordings']) or len(groups) != len(role['groups'])
                or clips != {x for x in universe if recording_group[x] in groups}
                or groups != {recording_group[x] for x in clips}):
            raise ValueError('A role must consist of whole exclusion groups')
        return clips

    check(options['excluded'])
    result = {}
    for option in PARTITIONS:
        roles = options[option]
        if set(roles) != {'calibration', 'confirmation'}:
            raise ValueError('Unexpected prospective role')
        cal, test = check(roles['calibration']), check(roles['confirmation'])
        if cal & test or (cal | test) & excluded or (cal | test | excluded) != universe:
            raise ValueError('Roles must disjointly cover the audited roster')
        result[option] = dict(calibration=cal, confirmation=test)
    if result['frozen_zero_shot']['calibration']:
        raise ValueError('Frozen transfer cannot reserve external calibration')
    return result


def aggregate(clips, rows, recording_group):
    per_group = defaultdict(lambda: dict(queries=0, eligible=0))
    keys = ('queries', 'with_targets', 'visible', 'eligible', 'masked_neighbors')
    value = {k: sum(rows[c][k] for c in clips) for k in keys}
    for c in sorted(clips):
        for k in ('queries', 'eligible'):
            per_group[recording_group[c]][k] += rows[c][k]
    sizes = [x['eligible'] for x in per_group.values()]
    return dict(recordings=len(clips), exclusion_groups=len(per_group), **value,
        min_eligible_group_rows=min(sizes) if sizes else None,
        max_eligible_group_rows=max(sizes) if sizes else None,
        largest_group_row_share=max(sizes)/sum(sizes) if sum(sizes) else None)


def stream(reader):
    """Read one prefix at a time. No calls to the label or predictor interfaces."""
    keys = ('queries', 'with_targets', 'visible', 'eligible', 'masked_neighbors')
    out = dict.fromkeys(keys, 0)
    chain = hashlib.sha256()
    for frame in reader.query_frames:
        raw = reader.get_inputs(int(frame))
        z = adapt_past(raw)
        ids, eligible, hist, mask = (z[k] for k in (
            'agent_id', 'target_eligible', 'history_xy', 'history_valid'))
        n = int(eligible.sum())
        if (not np.array_equal(ids, raw['agent_ids']) or n != int(mask.all(1).sum())
                or not np.isfinite(z['baseline_cv']).all() or not mask[:, -1].all()):
            raise ValueError('Past-only inventory or causal CV changed')
        out['queries'] += 1
        out['with_targets'] += int(n > 0)
        out['visible'] += len(ids)
        out['eligible'] += n
        out['masked_neighbors'] += len(ids)-n
        chain.update(np.asarray([frame, len(ids), n], dtype='<i8').tobytes())
        for item in (ids.astype('<i8', copy=False), hist.astype('<f8', copy=False),
                     mask, z['baseline_cv'].astype('<f8', copy=False)):
            chain.update(np.ascontiguousarray(item).tobytes())
    return out, chain.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('scan', 'replay'))
    phase = parser.parse_args().phase
    start = time.monotonic()
    proposal, constraint, inventory = map(read, (PROPOSAL, CONSTRAINT, CACHE))
    if (proposal['scientific_roles_assigned'] or proposal['outcome_readout_allowed']
            or proposal['predictive_admission'] or proposal['outcome_or_support_counts_used_for_partition']
            or proposal['risk_budget'] != .02 or not proposal['original_rules_unchanged']
            or (proposal['observation_steps'], proposal['prediction_steps'], proposal['raw_annotation_stride']) != (8, 12, 1)
            or proposal['exclusion_constraints_sha256'] != digest(CONSTRAINT)
            or proposal['cross_archive_analysis_sha256'] != digest(SCREEN)):
        raise ValueError('Require unchanged, label-blind, unassigned alternatives')
    groups = constraint['recording_group']
    proposed = role_sets(proposal['alternatives'], groups)
    meta = {x['sequence_id']: x for x in inventory['recordings']}
    if (set(meta) != set(groups) or len(meta) != len(inventory['recordings'])
            or len(meta) != 112 or len(set(groups.values())) != 45):
        raise ValueError('Require the entire screened archive')
    rows, chain = {}, hashlib.sha256()
    for i, (seq, info) in enumerate(sorted(meta.items()), 1):
        folder = DATA/'dronecrowd_recordings_v1'/seq
        if digest(folder/'metadata.json') != info['metadata_sha256']:
            raise ValueError('Changed recording metadata')
        reader = DroneCrowdSceneWindows(folder, PROBES[0], expected_xml_sha256=info['source_xml_sha256'],
            expected_provenance=inventory['provenance'])
        rows[seq], token = stream(reader)
        if rows[seq]['queries'] != info['main_query_frames']:
            raise ValueError('Query stream differs from cached boundary schedule')
        chain.update(seq.encode()+bytes.fromhex(token))
        if i % 16 == 0:
            print(dict(recordings=i, total=len(meta), seconds=time.monotonic()-start), flush=True)
    full = aggregate(set(meta), rows, groups)
    if full['queries'] != inventory['main_query_frames']:
        raise ValueError('Incomplete query scan')
    forecaster, controller = map(read, (FORECASTER_COVERAGE, CONTROLLER_COVERAGE))
    value = dict(prospective_alternatives={opt: {role: aggregate(ids, rows, groups)
        for role, ids in parts.items()} for opt, parts in proposed.items()},
        full=full, input_chain_sha256=chain.hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (PROPOSAL, CONSTRAINT, SCREEN,
            CACHE, FORECASTER_COVERAGE, CONTROLLER_COVERAGE)},
        cached_forecasters_screened=forecaster['covered_predictors'],
        controller_coverage_scope=controller.get('scope', 'two archived candidate cost heads only'),
        scientific_roles_assigned=False, learned_forecast_inference_run=False,
        cv_tensor_smoke_only=True, future_label_method_called=False, easy_reference_measured=False,
        independent_physical_sites_verified=False, original_rules_unchanged=True,
        observation_steps=8, prediction_steps=12, annotation_stride=1, coordinate_unit='image_pixel',
        stage5c_executed=False, smc_enabled=False, submission_ready=False)
    path = PUBLIC/'analysis.json'
    if phase == 'replay' and not path.is_file():
        raise ValueError('Run the full scan before replay')
    once(path, value)
    receipt = dict(phase=phase, seconds=time.monotonic()-start, analysis_sha256=digest(path),
        architecture=platform.machine(), queries=full['queries'], future_label_method_called=False)
    receipt_path = PUBLIC/f'{phase}_execution.json'
    if receipt_path.exists():
        if read(receipt_path)['analysis_sha256'] != receipt['analysis_sha256']:
            raise ValueError('Prior execution receipt binds a different scan')
    else:
        once(receipt_path, receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
