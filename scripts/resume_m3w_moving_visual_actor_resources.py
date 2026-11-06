"""Resume the frozen experiment with an explicit resource-only amendment."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT/'outputs/publication_readiness_2026_09/moving_visual_actor_v1/resource_amendment.json'


def runtime_settings(config, amendment):
    if config['disk_reserve_bytes'] != amendment['old_disk_reserve_bytes']:
        raise ValueError('Original resource policy changed')
    if config['artifact_cap_bytes'] != amendment['unchanged_artifact_cap_bytes']:
        raise ValueError('Artifact cap must not be relaxed')
    revised = copy.deepcopy(config)
    revised['disk_reserve_bytes'] = amendment['new_disk_reserve_bytes']
    return revised


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'verify', 'forecast', 'evaluate', 'replay'):
        raise ValueError('Resource amendment only continues the existing experiment')
    sys.path.insert(0, str(ROOT))
    amendment = json.loads(AMENDMENT.read_text())
    for relative, digest in amendment['unchanged_files'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed frozen scientific definition: '+relative)
    for path in (Path(__file__), AMENDMENT):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Commit the resource amendment before resuming')
    from scripts import run_m3w_moving_visual_actor as actor
    original = actor.runtime

    def amended_runtime(config, phase):
        print(json.dumps(dict(resource_amendment=str(AMENDMENT.relative_to(ROOT)),
            old_disk_reserve_bytes=amendment['old_disk_reserve_bytes'],
            new_disk_reserve_bytes=amendment['new_disk_reserve_bytes'],
            scientific_definitions_changed=False)), flush=True)
        return original(runtime_settings(config, amendment), phase)

    actor.runtime = amended_runtime
    actor.main()


if __name__ == '__main__':
    main()
