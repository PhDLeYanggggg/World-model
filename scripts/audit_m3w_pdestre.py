"""Audit 8/12 label support and shared IDs without downloading P-DESTRE videos."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

URL = "https://p-destre.di.ubi.pt/datasets/dataset.tar"
ETAG = '"9abb12e00-653fa65b6da3f"'
SIZE = 41_535_221_248
BLOCK = tarfile.BLOCKSIZE
ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/"external_data/PDESTRE_source_audit"
OUT = ROOT/"outputs/publication_readiness_2026_09/pdestre_source_intake_v1"
STRIDES = (1, 12)


def save(path, data):
    raw = json.dumps(data, indent=2, sort_keys=True)+"\n"
    if path.exists():
        if path.read_text() != raw:
            raise ValueError(f"Existing receipt changed: {path}")
        return
    tmp = path.with_suffix(".pending")
    tmp.write_text(raw)
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def digest(buf):
    return hashlib.sha256(buf).hexdigest()


def byte_range(start, length):
    end = start+length-1
    if not 0 <= start <= end < SIZE:
        raise ValueError("Range outside publisher snapshot")
    req = Request(URL, headers={"Range": f"bytes={start}-{end}", "If-Match": ETAG,
                                "Accept-Encoding": "identity", "User-Agent": "M3W research dataset audit"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as r:
                if (r.status != 206 or r.headers.get("ETag") != ETAG
                        or r.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"
                        or r.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError("Publisher did not serve the pinned exact byte range")
                b = r.read(length+1)
                if len(b) != length:
                    raise ValueError("Short or oversized publisher response")
                return b
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt == 2 or isinstance(e, HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2**attempt)


def directory(read, size, on_entry=lambda: None):
    """Read one verified tar header at a time and skip member payloads."""
    pos, out, seen = 0, [], set()
    while pos < size:
        header = read(pos, BLOCK)
        if header == b"\0"*BLOCK:
            if read(pos+BLOCK, BLOCK) != header:
                raise ValueError("Malformed tar terminator")
            return out
        t = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        p = PurePosixPath(t.name)
        if p.is_absolute() or ".." in p.parts or t.name in seen or not (t.isdir() or t.isreg()):
            raise ValueError(f"Unexpected archive member: {t.name}")
        seen.add(t.name)
        nxt = pos+BLOCK+((t.size+BLOCK-1)//BLOCK)*BLOCK
        if nxt > size or len(seen) > 1000:
            raise ValueError("Tar index exceeds expected bounds")
        if (t.isreg() and p.suffix.lower() in (".txt", ".mp4")
                and not p.name.startswith("._") and "__MACOSX" not in p.parts):
            out.append(dict(member=t.name, offset=pos+BLOCK, bytes=t.size, header_sha256=digest(header)))
        pos = nxt
        on_entry()
    raise ValueError("No tar terminator")


def pairs(entries):
    out = {}
    for e in entries:
        p = PurePosixPath(e["member"])
        kinds = out.setdefault(p.stem, {})
        kind = p.suffix.lower()
        if kind in kinds:
            raise ValueError(f"Ambiguous clip stem: {p.stem}")
        kinds[kind] = e
    if not out or any(set(v) != {".txt", ".mp4"} for v in out.values()):
        raise ValueError("Annotations and video roster do not match")
    return out


def identifiers(buf):
    frame, ident = [], []
    ignored = 0
    for line in io.BytesIO(buf):
        if not line.strip():
            continue
        # No pose, appearance, demographics or coordinate values are parsed.
        fields = line.split(b",", 2)
        if len(fields) != 3:
            raise ValueError("Expected publisher comma-separated frame and ID")
        t, i = int(fields[0]), int(fields[1])
        if t < 0 or i < -1:
            raise ValueError("Invalid identifier row")
        if i == -1:
            ignored += 1
        else:
            frame.append(t)
            ident.append(i)
    if not frame:
        raise ValueError("No identified rows")
    return np.asarray(frame), np.asarray(ident), ignored


def supports(tracks, stride):
    hist, complete, absent = Counter(), Counter(), Counter()
    end = max(t[-1] for t in tracks)
    for t in tracks:
        for r in np.unique(t % stride):
            f = t[t % stride == r]
            now = f[7:][(f[7:]-f[:-7] == 7*stride) & (f[7:]+12*stride <= end)]
            j = np.searchsorted(f, now[:, None]+stride*np.arange(1, 13))
            valid = (j < len(f)) & (f[j.clip(max=len(f)-1)] == now[:, None]+stride*np.arange(1, 13))
            hist.update(now.tolist())
            complete.update(now[valid.all(1)].tolist())
            absent.update(now[~valid.any(1)].tolist())
    queries = sorted(t for t, k in hist.items() if k >= 2)
    chosen, next_start = [], -1
    for t in queries:
        if t-7*stride >= next_start:
            chosen.append(t)
            next_start = t+12*stride+1
    n, c, a = (sum(d[t] for t in chosen) for d in (hist, complete, absent))
    return dict(overlapping_multi_agent_queries=len(queries), nonoverlapping_queries=len(chosen),
                first_selected_anchor=chosen[0] if chosen else None, history_targets=n, complete_targets=c,
                absent_targets=a, partial_targets=n-c-a,
                queries_with_two_complete=sum(complete[t] >= 2 for t in chosen),
                fully_labelled_queries=sum(complete[t] == hist[t] for t in chosen))


def audit_member(e, path):
    if path.exists():
        saved = load(path)
        if saved["source"] != e:
            raise ValueError("Stale annotation member receipt")
        return saved
    if not 0 < e["bytes"] <= 128_000_000:
        raise ValueError("Annotation member outside size guard")
    b = byte_range(e["offset"], e["bytes"])
    frame, ids, ignored = identifiers(b)
    order = np.lexsort((frame, ids))
    grouped = np.split(frame[order], np.flatnonzero(np.diff(ids[order]))+1)
    if any(np.any(np.diff(t) == 0) for t in grouped):
        raise ValueError("Duplicate (frame, ID) in released annotation")
    row = dict(source=e, sha256=digest(b), identified_rows=len(frame), unidentified_rows=ignored,
               first_frame=int(frame.min()), last_frame=int(frame.max()), n_tracks=len(grouped),
               strides={str(s): supports(grouped, s) for s in STRIDES})
    private = dict(row, ids=np.unique(ids).tolist())
    save(path, private)
    return private


def components(identities):
    groups = []
    for k, v in sorted(identities.items()):
        v = set(v)
        clips, rest = [k], []
        for ids, members in groups:
            if v & ids:
                v |= ids
                clips.extend(members)
            else:
                rest.append((ids, members))
        groups = [*rest, (v, clips)]
    return sorted((dict(clips=sorted(clips), tracks=len(ids)) for ids, clips in groups),
                  key=lambda x: (-len(x["clips"]), x["clips"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("index", "audit"))
    phase = p.parse_args().phase
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT/"range_index.json"
    if phase == "index":
        start = time.monotonic()
        entries = directory(byte_range, SIZE)
        clips = pairs(entries)
        first = clips[sorted(clips)[0]][".mp4"]
        b = byte_range(first["offset"], 32)
        if b[4:8] != b"ftyp":
            raise ValueError(f"Expected MP4 ftyp for {first['member']}; header {b[:12].hex()}")
        save(manifest, dict(url=URL, size=SIZE, etag=ETAG, clips=clips,
                            mp4_probe=dict(member=first["member"], bytes=len(b), sha256=digest(b)),
                            video_decoded=False, roles_assigned=False))
        print(f"Indexed {len(clips)} annotation/video pairs in {time.monotonic()-start:.2f}s.", flush=True)
        return
    index = load(manifest)
    if (index["url"], index["size"], index["etag"]) != (URL, SIZE, ETAG) or index["roles_assigned"]:
        raise ValueError("Wrong publisher snapshot or scientific role")
    rows, ids = {}, {}
    for k, v in sorted(index["clips"].items()):
        r = audit_member(v[".txt"], LOCAL/(digest(k.encode())+".json"))
        ids[k] = r.pop("ids")
        rows[k] = r
        print(f"Audited {len(rows)}/{len(index['clips'])} clips.", flush=True)
    groups = components(ids)
    totals = {str(s): {k: sum(r["strides"][str(s)][k] for r in rows.values()) for k in (
        "nonoverlapping_queries", "history_targets", "complete_targets", "absent_targets", "partial_targets",
        "queries_with_two_complete", "fully_labelled_queries")} for s in STRIDES}
    d = dict(index_sha256=digest(manifest.read_bytes()), archive_bytes=SIZE,
             annotation_bytes=sum(r["source"]["bytes"] for r in rows.values()), clips=len(rows),
             identified_rows=sum(r["identified_rows"] for r in rows.values()),
             unidentified_rows=sum(r["unidentified_rows"] for r in rows.values()),
             unique_release_ids=len(set().union(*map(set, ids.values()))),
             shared_id_components=groups, stride_totals=totals, recordings=rows,
             role="source qualification only", roles_assigned=False, video_decoded=False,
             no_trajectories_or_biometrics_parsed=True, forecast_scoring=False, reserved_outcomes_opened=False)
    save(OUT/"analysis.json", d)
    print(json.dumps({k: d[k] for k in ("clips", "stride_totals", "unique_release_ids")}, indent=2))


if __name__ == "__main__":
    main()
