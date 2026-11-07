import numpy as np
import pytest

from scripts import benchmark_m3w_native_cpu_shards as bench
from scripts import run_m3w_dronecrowd_confirmation_parallel as exec3


def test_workload_shards_are_exhaustive_disjoint_and_deterministic():
    work = {'a': 8, 'b': 4, 'c': 3, 'd': 2, 'e': 1}
    result = exec3.assign(list(work), work)
    assert sorted(k for shard in result for k in shard) == sorted(work)
    assert exec3.assign(list(reversed(work)), work) == result
    with pytest.raises(ValueError):
        exec3.assign(['a'], work)


def test_dev_prefix_rejects_cross_recording_and_missing_frame():
    d = dict(groups=[np.array([0, 1]), np.array([2])], recordings=np.array([4, 4, 5]),
             history=np.zeros((3, 8, 2)), agents=np.array([1, 2, 3]))
    frames = np.array([8, 8, 8])
    out = bench.PrefixSource(d, frames, '4', [8]).get_inputs(8)
    np.testing.assert_array_equal(out['agent_ids'], [1, 2])
    assert out['observation_frames'][-1] == 8
    with pytest.raises(ValueError):
        bench.PrefixSource(d, frames, '4', [9])
