"""Read official Waterloo annotation archives without allocating scientific roles."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_waterloo_intake import audit_db, sha

ASSETS = ROOT/"external_data/Waterloo_WISE_annotations"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09/waterloo_annotations_v1"
SOURCES = {
    "crosswalk": dict(bytes=64691909, files=17, site="Ring Road campus crosswalk",
        url="https://wiselab.uwaterloo.ca/multi-agent-pedestrian-traffic/crosswalk_dbfiles.zip"),
    "intersection": dict(bytes=610803071, files=14, site="University Avenue and Weber Street",
        url="https://wiselab.uwaterloo.ca/multi-agent-intersection-traffic/intersection_dbfiles.zip"),
}


def write(path, result):
    value = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if path.exists():
        if path.read_text() != value:
            raise ValueError("Audit replay differs from saved result")
        return
    tmp = path.with_suffix(".pending")
    tmp.write_text(value)
    tmp.replace(path)


def audit():
    report = {}
    for site, meta in SOURCES.items():
        archive = ASSETS/f"{site}_dbfiles.zip"
        if archive.stat().st_size != meta["bytes"]:
            raise ValueError("Archive length differs from official source response")
        records = []
        with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=ASSETS) as temp:
            members = sorted(z.infolist(), key=lambda x:x.filename)
            if len(members) != meta["files"] or len({m.filename for m in members}) != len(members):
                raise ValueError("Unexpected member count or duplicate names")
            for member in members:
                if member.file_size > 850*1024**2 or shutil.disk_usage(ASSETS).free < 2*1024**3:
                    raise ValueError("Unexpected member size or insufficient scratch")
                if Path(member.filename).name != member.filename or not member.filename.endswith(".db"):
                    raise ValueError("Expected database-only archive")
                local = Path(temp)/"member.db"
                with z.open(member) as src, local.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024**2)
                records.append(dict(audit_db(local, site, member.filename), member_sha256=sha(local)))
                local.unlink()
                print(json.dumps(dict(pid=os.getpid(), site=site, member=member.filename,
                                      status=records[-1]["support_status"])), flush=True)
        totals = {s: {k: Counter() for k in ("all","pedestrian")} for s in ("1","12")}
        admitted = [r for r in records if r["availability"] is not None]
        for row in admitted:
            for s, strata in row["availability"].items():
                for key, counts in strata.items():
                    totals[s][key].update(counts)
        report[site] = dict(source=meta, archive_sha256=sha(archive), recordings=records,
            aggregate=dict(trajectory_rows=sum(r["rows"] for r in records),
                track_ids=sum(r["track_ids"] for r in records), pedestrian_tracks=sum(r["pedestrian_tracks"] for r in records),
                pedestrian_rows=sum(r["pedestrian_rows"] for r in records), recordings=len(records),
                aligned_recordings=len(admitted), supported_recording_sums=totals))
    return dict(sites=report, result_source="fresh_run", distinct_publisher_site_locations=2,
                source_coordinate="publisher-described UTM zone 17T; geometry not independently validated",
                time_scope="exported timestamp lattice only; no source-to-M3W duration equivalence",
                producer_temporal_provenance="not_verified", sources_formally_admitted=False,
                roles_assigned=False, forecast_errors_evaluated=False, dronecrowd_confirmation_access=False,
                no_change_to_original_protocol=True, implementation={str(p.relative_to(ROOT)):sha(p)
                    for p in (Path(__file__), ROOT/"src/evaluation/m3w_waterloo_intake.py")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    t = time.monotonic()
    if args.replay and not (PUBLIC/"analysis.json").is_file():
        raise FileNotFoundError("Score the intake before replay")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write(PUBLIC/"analysis.json", audit())
    write(PUBLIC/("replay.json" if args.replay else "execution.json"),
          dict(pid=os.getpid(), seconds=time.monotonic()-t, analysis_sha256=sha(PUBLIC/"analysis.json")))
