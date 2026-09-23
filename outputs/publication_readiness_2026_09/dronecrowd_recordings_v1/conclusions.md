# DroneCrowd Lazy Recording Cache

## Result

`fresh_run`: all 112 XML recordings were converted and compared element-for-element
with the original parsed source. The cache contains 20,800 recording-local tracks,
4,864,280 visible/nonoccluded positions and 112,543,744 array bytes (107.33 MiB).
Positions remain float64 head-box centers in image pixels. Invisible padded rows
remain masked; neither bounds clipping nor interpolation was introduced.

The reader exposes 31,472 shared query times for eight observed and twelve future
stride-1 annotation steps. It does not materialize per-agent episodes. Query
scheduling uses clip boundaries, not future agent visibility. Currently visible
agents, including those with incomplete past histories or missing future labels,
remain in inputs. Inputs and supervision are separate APIs; backward velocity is
invalid across any raw visibility gap. The API can also read inputs at the end
of a clip when future labels are unavailable.

All 336 real cached-prefix/source-prefix comparisons and future-truncation checks
pass. These are new cache integration checks, not a claim of recovered annotation
interpolation provenance. The source-time causality limitation is unchanged.

## Verification and Cost

Final conversion took 39.26 seconds, including 37.40 seconds of XML parsing and
0.895 seconds opening and hash-checking all 112 mmap caches. Constructing the
336 checked scene inputs took 0.070 seconds. These are component timings during
local source auditing, **not** training throughput or a controlled end-to-end
speedup measurement. Future loaders can avoid repeated XML parsing.

A separate `--resume` run re-parsed the original XML, rechecked every array and
reconstructed the identical analysis in 38.64 seconds. It wrote zero new caches;
its result is `cached_verified`, not new conversion or training. Analysis SHA256:
`1bb9946fa1bb62c720b40acb5aa7348dc4e8bf852a44c0d35b8765f36e98e9be`.

The first attempt exposed a JSON tuple/list equality error during report replay,
not a coordinate discrepancy. It is retained locally under the ignored
`dronecrowd_recordings_v1_attempt1` directory. The final schema uses JSON-stable
lists; failed writes cannot publish a completed recording directory. Both
regressions now have tests. Eleven cache tests and 222 scoped source, reader,
split-contract and admission tests pass. This is not the complete legacy suite.

## Admission Boundary

All recordings remain `unassigned_quarantine`. This cache is deliberately not
an admission receipt and cannot be substituted for a reviewed experimental
contract. It supplies no physical-site identities, model predictions, goals,
teacher outputs, normalizers, training or confirmation results. Site exclusion
constraints and scientific roles still need to be bound before forecasting.

The task is offline forecasting from exported annotations. Native stride-1 steps
are not physical-time equivalents of SDD stride-12 steps. Effective seconds and
metric geometry are unverified; this is not true 3D or a foundation-model result.
Stage5C and SMC remain off. Raw/cache arrays stay outside Git.

## Reproduce

From the repository root, after acquiring the hash-bound official annotation
archive and retaining its intake reports:

```sh
.venv-pytorch/bin/python scripts/build_m3w_dronecrowd_recording_cache.py
.venv-pytorch/bin/python scripts/build_m3w_dronecrowd_recording_cache.py --resume
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_cache.py tests/test_m3w_dronecrowd_windows.py
```

The first command is for a new cache only. An existing complete cache is verified,
never silently overwritten. Interrupted per-recording writes stay in separate
pending directories; the next run can rebuild the unpublished recording.
