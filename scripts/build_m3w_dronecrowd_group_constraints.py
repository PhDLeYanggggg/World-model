"""Freeze conservative recording-exclusion groups, not independent scene roles."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_unification.m3w_dronecrowd_cache import digest_file
from src.evaluation.m3w_dronecrowd_grouping import group_evidence, require_no_cross_role_edges


def main():
    base = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_grouping_v2"
    analysis_path = base / "analysis.json"
    evidence = json.loads(analysis_path.read_text())
    execution = json.loads((base / "execution.json").read_text())
    if digest_file(analysis_path) != execution["analysis_sha256"] or evidence["clip_pairs_complete"] != 6216:
        raise ValueError("Complete checksum-verified multi-view audit required")
    identity = json.loads((base / "run_identity.json").read_text())
    image_manifest = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1/source_manifest.json"
    if digest_file(image_manifest) != identity["image_manifest_sha256"]:
        raise ValueError("Image reference changed")
    images = json.loads(image_manifest.read_text())["frames"]
    by_id = {(r["scene_id"], r["frame_one_based"]): r for r in images}
    roles = {r["scene_id"]: r["release_split"] for r in images}
    config_path = ROOT / "configs/m3w_dronecrowd_visual_constraints_v1.json"
    config = json.loads(config_path.read_text())
    manual = []
    for row in config["reviewed_pairs"]:
        bound = {}
        if row["left"] == row["right"] or row["relation"] not in {
            "visually_corroborated_overlap", "possible_shared_site_do_not_separate"
        }:
            raise ValueError("Invalid review constraint")
        for node in (row["left"], row["right"]):
            image = by_id[node, 1]
            if digest_file(ROOT / "external_data/DroneCrowd_images" / image["local_name"]) != image["sha256"]:
                raise ValueError("Reviewed image changed")
            bound[node] = {"frame_one_based": 1, "image_sha256": image["sha256"]}
        manual.append({**row, "status": "ambiguous", "source_frames": bound})
    pairs = evidence["supported_pairs"] + manual
    groups = group_evidence(sorted(roles), pairs, evidence["prior_positive_edges_preserved"])
    components = groups["conservative_components"]
    assignments = {node: f"exclusion_{i:03d}" for i, group in enumerate(components) for node in group}
    require_no_cross_role_edges(assignments, pairs, evidence["prior_positive_edges_preserved"])
    try:
        require_no_cross_role_edges(roles, pairs, evidence["prior_positive_edges_preserved"])
    except ValueError as exc:
        release_rejection = str(exc)
    else:
        raise ValueError("Known release split overlap unexpectedly disappeared")
    mixed = [c for c in components if len({roles[n] for n in c}) > 1]
    value = {
        "result_source": "fresh_run", "scope": "no_cross_role_exclusion_constraints_not_independent_sites",
        "bindings": {"multi_view_analysis_sha256": digest_file(analysis_path),
                     "source_manifest_sha256": digest_file(image_manifest),
                     "visual_review_sha256": digest_file(config_path),
                     "script_sha256": digest_file(Path(__file__)),
                     "grouping_module_sha256": digest_file(ROOT / "src/evaluation/m3w_dronecrowd_grouping.py")},
        "manual_constraints": manual, "recording_group": assignments,
        "conservative_components": components, "component_count": len(components),
        "cross_release_components": mixed, "cross_release_component_count": len(mixed),
        "original_release_split_rejected": release_rejection,
        "constraint_check_passed": True, "negative_matches_establish_independence": False,
        "physical_site_count": None, "scientific_roles_assigned": False,
        "all_recordings_status": "unassigned_quarantine",
        "forecast_errors_read": False, "training": "not_run", "stage5c_executed": False, "smc_enabled": False,
    }
    path = base / "exclusion_constraints.json"
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Existing constraint manifest changed; do not overwrite")
    path.write_text(payload)
    print(json.dumps({"constraint_manifest_sha256": digest_file(path), "component_count": len(components),
                      "cross_release_component_count": len(mixed), "physical_site_count": None}))


if __name__ == "__main__":
    main()
