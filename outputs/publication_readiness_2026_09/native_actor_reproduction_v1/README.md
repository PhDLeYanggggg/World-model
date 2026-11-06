# Standalone Native Actor Aggregate Evidence

I packaged the original primary results of the three completed native-stride
actor experiments so their numerical reporting can be checked without my
workspace. All three negative component results remain negative. This is an
aggregate-evidence artifact,not a neural model or full training release.

## What Was Verified

Fresh_run:deterministic253,746byte archive with six files;two clean isolated
executions exit0. Python runs with isolated mode and site-packages disabled,
a temporary home,and file/network/subprocess audit guards. Only extracted
artifact and interpreter files may be read. The guard is not an OS sandbox.
Two reconstructed outputs match the frozen expected hashes. Five focused
checks pass0.26s,including missing-source/undefined-value preservation and
detection of a changed original bootstrap seed.

Cached_verified:three pinned public scalar analyses,not raw targets or neural
forecasts. The exporter replays all fourteen source means and the original
3,000draw indices. Prefix-loss uses seed39017;the two image experiments use
39107. Those distinct registered draws are retained,not silently pooled.
Original1/3/6/12prefixes and known/partial/wholly-unknown counts remain.

Direct local/author/repository markers are removed with an allowlist and
fixed scan. Neutral source aliases keep the original order and weights.
Recognition from already-public results is still possible;anonymity is not
certified and no submission is performed.

## Rebuild And Recheck

From the repository with NumPy installed:

```sh
.venv-pytorch/bin/python scripts/build_m3w_native_actor_reproduction.py
.venv-pytorch/bin/python scripts/build_m3w_native_actor_reproduction.py --verify
.venv-pytorch/bin/python -m pytest tests/test_native_actor_portable.py
```

The first command refuses to overwrite an existing archive;use `--verify` for
the already-built artifact. Local archive:
`data/stage_cvpr2027_experiments/native_actor_reproduction_v1/evidence_bundle.zip`.
It is not a raw-data/cache/checkpoint redistribution and is not committed.
After extraction,the standalone command is:

```sh
python3 -I -S reproduce.py --verify --output reproduced
```

No NumPy,Torch,network or repository is needed inside the extracted artifact.
The scope/readme template is also tracked under
`reproducibility/native_actor_evidence/README.md`.

## What Remains Missing

Not_run:raw-data reconstruction,checkpoint/forecast replay inside this bundle,
complete model-training reproduction,independent scientific confirmation and
the full original selected-policy risk contract. The live operator-factor
experiment is not included or prematurely scored. This artifact does not
make the manuscript submission-ready. Native observation8/prediction12,
stride1,image coordinates remain distinct from the older stride12and SDD
raw-horizon studies. No verified seconds,meters,true3D,foundation,human-gold
or physical-safety claim. Stage5C execution and SMC remain off.
