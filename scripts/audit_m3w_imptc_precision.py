"""Fixed precision repair on input-only probes; old failed evidence is retained."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
if platform.system()=="Darwin" and platform.machine()!="arm64": raise RuntimeError("Native arm64 required")
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(k,"2")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.data_unification.m3w_quantized_prefix import QuantizedPrefixAdapter
from src.data_unification.m3w_unit_free_prefix import unit_free_cost_features
from scripts.audit_m3w_imptc_input_contract import load_models, infer, compare
from scripts.run_m3w_native_forecast import immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest


def read(p): return json.loads((ROOT/p).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--verify",action="store_true");args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    old=ROOT/"data/stage_cvpr2027_experiments/imptc_input_contract_v1"
    pub=ROOT/"outputs/publication_readiness_2026_09/imptc_input_contract_v1"
    out=ROOT/"data/stage_cvpr2027_experiments/imptc_precision_v2"
    report=ROOT/"outputs/publication_readiness_2026_09/imptc_precision_v2"
    analysis=read(pub/"analysis.json");identity=read(old/"identity.json")
    assert file_digest(old/"identity.json")==analysis["identity_sha256"]
    for p,h in identity["code_hashes"].items(): assert file_digest(ROOT/p)==h
    lineage=read(pub/"producer_chain.json")
    assert file_digest(pub/"producer_chain.json")==analysis["producer_chain_sha256"]
    for r in lineage["receipts"]: assert file_digest(ROOT/r["checkpoint"])==r["checkpoint_sha256"]
    source=read(identity["config"]["source_audit"])
    assert file_digest(ROOT/identity["config"]["source_audit"])==identity["config"]["source_audit_sha256"]
    for r in source["cache_files"]: assert file_digest(ROOT/r["path"])==r["sha256"]
    paths=("scripts/audit_m3w_imptc_precision.py","src/data_unification/m3w_quantized_prefix.py","tests/test_m3w_quantized_prefix.py",str((report/"registration.md").relative_to(ROOT)))
    new_identity=dict(source_analysis_sha256=file_digest(pub/"analysis.json"),
        prefix_selection_sha256=file_digest(old/"prefix_selection.json"), code_hashes={p:file_digest(ROOT/p) for p in paths},
        coordinate_factors=[.01,1.,100.],dimensionless_quantum=1e-9,torch_threads=2,interop_threads=1)
    immutable_json(out/"identity.json",new_identity)
    if args.verify and not (report/"analysis.json").exists(): raise ValueError("No original run")
    support=read(old/"prefix_selection.json");groups={f:[] for f in (.01,1.,100.)};cache={}
    for query in support["queries"]:
        seq,frame=query["sequence"],query["frame"]
        if seq not in cache: cache[seq]=np.load(ROOT/f"data/stage_cvpr2027_experiments/imptc_intake_v1/{seq}_rows.npy",allow_pickle=False,mmap_mode="r")
        rows=cache[seq];b=rows[(rows["frame_id"]>=frame-7)&(rows["frame_id"]<=frame)]
        p=np.column_stack((b["frame_id"],b["agent_id"],b["x"],b["y"]))
        for factor in groups:
            changed=p.copy();changed[:,2:]*=factor
            g,s=QuantizedPrefixAdapter(changed,query_frame=frame,recording_id=seq).geometry_batch()
            assert [a["agent_id"] for a in s["agents"]]==query["agents"]
            groups[factor].append(g)
    models,_=load_models(lineage);arrays={};bank={}
    for factor,parts in groups.items():
        g=np.concatenate(parts);fields=dict(geometry=g)
        for name,model in models.items():
            p=infer(model,g);fields[name+"_prediction"]=p
            fields[name+"_cost_features"]=unit_free_cost_features(g,p)[0]
        bank[factor]=fields
        arrays.update({str(factor).replace(".","p")+"_"+k:v for k,v in fields.items()})
    differences={str(f):{k:compare(bank[1.][k],v) for k,v in bank[f].items()} for f in (.01,100.)}
    previous=analysis["private_arrays"];assert file_digest(ROOT/previous["path"])==previous["sha256"]
    with np.load(ROOT/previous["path"],allow_pickle=False) as z:
        old_g=z["unit_free_1p0_geometry"]
        old_moving=np.any(old_g[:,14:16]!=old_g[:,12:14],axis=1)
    g=bank[1.]["geometry"];new_moving=np.any(g[:,14:16]!=g[:,12:14],axis=1)
    write_arrays(out/"probes.npz",arrays)
    result=dict(result_source="fresh_run",identity_sha256=file_digest(out/"identity.json"),
        windows=len(g),queries=len(support["queries"]),differences=differences,
        measured_normalized_tolerance_pass=all(v["rows_over_1e_5"]==0 for r in differences.values() for v in r.values()),
        moving_to_static=int((old_moving & ~new_moving).sum()),static_to_moving=int((~old_moving & new_moving).sum()),
        original_failed_gate_unchanged=True,earlier_code_changed=False,fitting=False,future_error_readout=False,
        deployment=False,stage5c_executed=False,smc_enabled=False,lossless=False,
        private_arrays=dict(path=str((out/"probes.npz").relative_to(ROOT)),sha256=file_digest(out/"probes.npz")))
    immutable_json(report/"analysis.json",result)
    if args.verify: immutable_json(report/"verification.json",dict(exact=True,analysis_sha256=file_digest(report/"analysis.json")))
    print(json.dumps(result))


if __name__=="__main__":main()
