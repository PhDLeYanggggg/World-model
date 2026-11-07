"""Engineering smoke test for every cached external recording, no outcome readout."""
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m3w_dronecrowd_cross_archive import DATA, HOME, digest, once, read
from src.data_unification.m3w_dronecrowd_cache import DroneCrowdSceneWindows
from src.data_unification.m3w_dronecrowd_frozen_inputs import adapt_past
from src.evaluation.m3w_dronecrowd_windows import PROBES
from src.world_model.m3w_european_source_forecast import pack_scene
from src.world_model.m3w_native_forecast import pack_geometry


def main():
    start = time.monotonic()
    path = HOME/'dronecrowd_recordings_v1/analysis.json'; meta = read(path); rows = []
    for item in meta['recordings']:
        node = item['sequence_id']; folder = DATA/'dronecrowd_recordings_v1'/node
        if digest(folder/'metadata.json') != item['metadata_sha256']:
            raise ValueError('Changed source recording metadata')
        reader = DroneCrowdSceneWindows(folder, PROBES[0], expected_xml_sha256=item['source_xml_sha256'],
            expected_provenance=meta['provenance'])
        # Use only clip length to locate three past prefixes. Never call get_labels.
        q = reader.query_frames; inspected = []
        for frame in (q[0], q[len(q)//2], q[-1]):
            inputs = adapt_past(reader.get_inputs(int(frame)))
            g, selected = pack_scene(inputs)
            tensors = pack_geometry(g)
            if len(selected) != int(inputs['target_eligible'].sum()) or tensors['history'].shape != (len(selected), 8, 3):
                raise AssertionError('Causal target or tensor schema changed')
            inspected.append(dict(frame=int(frame), visible=len(inputs['agent_id']), eligible=len(selected)))
        rows.append(dict(sequence=node, query_frames=len(q), inspected=inspected))
    value = dict(recordings=len(rows), prefix_queries=3*len(rows), records=rows, source_manifest_sha256=digest(path),
        adaptation_sha256=digest(Path(__file__).resolve().parents[1]/'src/data_unification/m3w_dronecrowd_frozen_inputs.py'),
        label_method_called=False, recording_array_hashes_checked=True, forecast_inference_run=False,
        new_predictive_admission=False, scientific_roles_assigned=False, independent_confirmation=False,
        stage5c_executed=False, smc_enabled=False, submission_ready=False)
    out = HOME/'dronecrowd_frozen_inputs_v1/analysis.json'; once(out, value)
    print(dict(recordings=len(rows), prefix_queries=value['prefix_queries'], seconds=time.monotonic()-start,
        sha256=digest(out)), flush=True)


if __name__ == '__main__':
    main()
