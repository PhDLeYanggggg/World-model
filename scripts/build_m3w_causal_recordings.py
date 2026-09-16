"""Rebuild canonical raw-position windows; do not assign an official split or train."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_unification.m3w_causal_recordings import build_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m3w_publication_recordings.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/stage_cvpr2027_causal")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/publication_readiness_2026_09")
    args = parser.parse_args()
    manifest = build_catalog(ROOT, args.config, args.cache_dir)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "causal_recording_rebuild.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Canonical Causal Recording Rebuild", "",
             "Fresh raw-position conversion. No legacy teacher or learned cache reused; no training or new performance result.", "",
             f"- Canonical recordings used: {manifest['canonical_recordings_used']}",
             f"- Physical scene groups: {manifest['physical_scene_groups']}",
             "- A collection label such as TrajNet is not an independent dataset domain.",
             "- Named aliases are conservatively co-grouped; only byte-equal aliases are proven numerically identical.",
             "- One incomplete challenge excerpt group is quarantined. No missing future labels are imputed.", "",
             "| recording | scene group | agents | raw points | 8 observed / 12 future steps | raw10 | raw25 | raw50 | raw100 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in manifest["recordings"]:
        c = r["raw_exact_windows"]
        lines.append(f"| {r['id']} | {r['physical_scene']} | {r['agents']} | {r['points']} | {r['observation_step_windows']} | "
                     f"{c['10']} | {c['25']} | {c['50']} | {c['100']} |")
    lines += ["", "## Boundaries", "",
              "Both protocols are candidate data views, not an approved official benchmark. Observation steps are not "
              "seconds. Exact raw horizons require a real label at that frame and a continuous annotation sequence; "
              "there is no nearest-future-frame substitution or interpolation.", "",
              "Past histories, current visible neighbors and causal baseline rollouts are loaded lazily from uncompressed "
              "npy arrays. Labels have a separate reader. No central velocity, future endpoint, future availability mask, "
              "remaining track length, old selector output, future goal or test normalization enters the input schema.", "",
              "No scene image, goal map, latent representation or learned dynamics has been added here. Neighbor geometry "
              "is observed trajectory context, not evidence of multimodal or physical-world success.", "",
              "The raw-frame and observed-step views overlap and must never be split independently by rows. "
              "Each recording and its aliases must stay together. A strict scene split also keeps Zara01/02/03 together "
              "and University recordings together, unlike a merely file-level split.", "",
              "The local documentation contains unit, FPS and homography claims, but it has not been calibrated against "
              "each selected representation here. Dataset-local coordinates and raw-frame/step claims are retained.", "",
              "The old exposed datasets are development material. No new untouched confirmation set is claimed. "
              "Protocol selection, train-only fitting, strong-baseline comparison and independent confirmation remain pending."]
    (args.report_dir / "causal_recording_rebuild.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"canonical_recordings": manifest["canonical_recordings_used"],
                      "raw_points": sum(r["points"] for r in manifest["recordings"]),
                      "observation_step_windows": sum(r["observation_step_windows"] for r in manifest["recordings"]),
                      "raw_exact_windows": {h: sum(r["raw_exact_windows"][h] for r in manifest["recordings"]) for h in ("10", "25", "50", "100")}}, indent=2))


if __name__ == "__main__":
    main()
