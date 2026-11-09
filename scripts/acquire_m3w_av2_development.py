"""Extend the fixed AV2 publisher-train UUID sample without inspecting labels."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlencode
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_native_trajectory as av

base, study = av.base, av.study
SOURCE = av.ROOT/"external_data/Argoverse2_mf_train_expansion_v1"
OUT = av.OUT.parent/"av2_data_scale_v1"
HEX, PER_PREFIX = "0123456789abcdef", 64
MAX_TOTAL = 350_000_000


def listing(raw, prefix):
    xml = ElementTree.fromstring(raw)
    out = {}
    for entry in xml.findall("{*}Contents"):
        key = entry.findtext("{*}Key")
        match = av.KEY.fullmatch(key or "")
        if not match:
            continue
        name = match[1]
        if not name.startswith(prefix) or name in out:
            raise ValueError("Unexpected or duplicated UUID in publisher prefix")
        size, etag = int(entry.findtext("{*}Size")), entry.findtext("{*}ETag")
        if not 0 < size < 4_000_000 or not etag:
            raise ValueError("Unexpected publisher object receipt")
        out[name] = dict(key=key, bytes=size, etag=etag)
    return out, xml.findtext("{*}NextContinuationToken")


def sample(prefix):
    out, token = {}, None
    while len(out) < PER_PREFIX:
        q = {"list-type": 2, "prefix": av.PREFIX+prefix, "max-keys": 256}
        if token:
            q["continuation-token"] = token
        raw, _ = av.get("?"+urlencode(q), 300_000)
        values, token = listing(raw, prefix)
        if out.keys() & values.keys():
            raise ValueError("Listing pages repeated UUIDs")
        out.update(values)
        if not token:
            break
    if len(out) < PER_PREFIX:
        raise ValueError("Incomplete UUID prefix")
    return {k: out[k] for k in sorted(out)[:PER_PREFIX]}


def register():
    old_path = av.SOURCE/"manifest.json"
    old = study.read(old_path)["objects"]
    p = OUT/"data_registration.json"
    base.immutable(p, dict(source=av.HOST+av.PREFIX, official_docs="https://argoverse.github.io/user-guide/tasks/motion_forecasting.html",
        previous=study.artifact(old_path), code=study.artifact(Path(__file__)),
        select="first 64 lexicographic scenario UUIDs per first hex digit; previous first 8 retained",
        archive="AV2 publisher train only; no maps, validation or test", size_cap_bytes=MAX_TOTAL,
        no_future_selection=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), old


def fetch(name, info, old):
    match = av.KEY.fullmatch(info["key"])
    if match is None or name != match[1]:
        raise ValueError("The download is not a registered AV2 train scenario")
    if name in old:
        r = old[name]
        if info != r["object"]:
            raise ValueError("Previous publisher object changed")
        study.checked(r["local"])
        return name, r
    p = SOURCE/f"{name}.parquet"
    receipt = p.with_suffix(".json")
    if receipt.exists():
        r = study.read(receipt)
        if r["object"] != info or study.checked(r["local"]) != p:
            raise ValueError("Expansion download no longer matches its receipt")
        return name, r
    content, etag = av.get(info["key"], info["bytes"])
    if len(content) != info["bytes"] or etag != info["etag"] or content[:4] != b"PAR1":
        raise OSError("Downloaded payload does not match publisher listing")
    tmp = p.with_suffix(".pending")
    tmp.write_bytes(content)
    tmp.replace(p)
    r = dict(object=info, local=study.artifact(p))
    base.immutable(receipt, r)
    return name, r


def acquire(key, old):
    selected = SOURCE/"selection.json"
    if not selected.exists():
        with ThreadPoolExecutor(max_workers=6) as pool:
            objects = {k: v for group in pool.map(sample, HEX) for k, v in group.items()}
        base.immutable(selected, dict(registration=key, objects=objects))
    pick = study.read(selected)
    objects = pick["objects"]
    if (pick["registration"] != key or not set(old) <= set(objects)
            or Counter(k[0] for k in objects) != dict.fromkeys(HEX, PER_PREFIX)
            or sum(v["bytes"] for v in objects.values()) > MAX_TOTAL):
        raise ValueError("Unexpected registered expansion roster or footprint")
    receipts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, r in pool.map(lambda p: fetch(*p, old), sorted(objects.items())):
            receipts[n] = r
            if len(receipts) % 32 == 0:
                print(f"{len(receipts)}/{len(objects)} train objects verified", flush=True)
    base.immutable(SOURCE/"manifest.json", dict(registration=key, selection=study.artifact(selected), objects=receipts))
    print(f"{len(receipts)-len(old)} added; {len(old)} verified reused", flush=True)


def main():
    subprocess.run(["git", "check-ignore", "--quiet", str(SOURCE/"manifest.json")], cwd=av.ROOT, check=True)
    if any(p.is_symlink() for p in (SOURCE, *SOURCE.parents)):
        raise ValueError("Refuse symlinked source directory")
    for d in (SOURCE, OUT):
        d.mkdir(parents=True, exist_ok=True)
    with (SOURCE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquire(*register())


if __name__ == "__main__":
    main()
