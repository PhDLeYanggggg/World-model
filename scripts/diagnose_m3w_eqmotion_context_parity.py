"""Read-only input-packet diagnosis for the failed frozen-context pilot."""
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_eqmotion_context_cache as cache
from src.world_model.m3w_frozen_interaction_capture import InteractionCapture
from src.world_model.m3w_native_forecast import pack_geometry


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    cfg, architecture, designs, current = cache.admission()
    directory = cache.parent.forecaster.source.PRIVATE / 'packed'
    receipt = json.loads((directory / 'receipt.json').read_text())
    path = directory / 'geometry.npy'
    if cache.sha(path) != receipt['arrays']['geometry']:
        raise ValueError('Original forecaster geometry changed')
    original = np.load(path, mmap_mode='r', allow_pickle=False)
    for field in ('sites', 'recordings', 'frames', 'agents', 'history', 'origin'):
        np.testing.assert_array_equal(current[field], np.load(directory / (field + '.npy'), mmap_mode='r'))
    fields = dict(ego_history_xy=(0, 16), ego_history_time=(16, 24),
                  neighbor_xy=(38, 166), neighbor_time=(166, 230),
                  neighbor_mask=(230, 294), baseline=(332, 356),
                  all_geometry=(0, 476))
    differences = {}
    for name, (lo, hi) in fields.items():
        count = elements = 0
        maximum = 0.
        for a in range(0, len(original), 4096):
            x = original[a:a+4096, lo:hi]
            y = current['geometry'][a:a+4096, lo:hi]
            count += int(np.any(x != y, axis=1).sum())
            elements += int(np.count_nonzero(x != y))
            maximum = max(maximum, float(np.max(np.abs(x - y))))
        differences[name] = dict(rows_different=count, elements_different=elements, max_absolute_difference=maximum)
    cells = []
    for design in designs:
        model = cache.load_model(architecture, design)
        ids = design['full_ids']
        n = (len(ids) + 127) // 128
        with np.load(cache.checked(design['forecast']), allow_pickle=False) as z:
            expected = z['prediction']
            with InteractionCapture(model) as capture:
                for offset in sorted({0, n//2*128, (n-1)*128}):
                    batch = ids[offset:offset+128]
                    a = pack_geometry(current['geometry'][batch])
                    b = pack_geometry(original[batch])
                    rng = torch.get_rng_state().clone()
                    p, _ = capture.run(a)
                    q, _ = capture.run(b)
                    with torch.no_grad():
                        plain_a, plain_b = model(a), model(b)
                    torch.testing.assert_close(p, plain_a, rtol=0, atol=0)
                    torch.testing.assert_close(q, plain_b, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    target = expected[offset:offset+128]
                    np.testing.assert_array_equal(q.numpy(), target)
                    cell = dict(trial=design['trial'], offset=offset, rows=len(batch),
                                current_input_prediction_exact=bool(np.array_equal(p.numpy(), target)),
                                current_input_mismatched_elements=int(np.count_nonzero(p.numpy() != target)),
                                current_input_max_absolute_error=float(np.max(np.abs(p.numpy() - target))),
                                original_input_prediction_exact=True, passive_hook_exact=True, RNG_exact=True)
                    cells.append(cell)
                    print(json.dumps(cell), flush=True)
    out = dict(result_source='fresh_run', pid=os.getpid(), seconds=time.monotonic()-start,
               original_pilot_failed=True, original_pilot_log=cache.artifact(cache.PRIVATE/'pilot.log'),
               original_geometry=cache.artifact(path), current_geometry_shape=list(current['geometry'].shape),
               row_identity_fields_exact=True, geometry_differences=differences, cells=cells,
               all_original_input_predictions_exact=True, future_labels_used=False,
               new_optimizer_updates=0, feature_utility_evaluated=False,
               deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    cache.once(cache.PUBLIC/'input_parity_diagnosis.json', out)


if __name__ == '__main__':
    main()
