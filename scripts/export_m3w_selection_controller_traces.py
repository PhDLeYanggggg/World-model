"""Export only scalar loss traces from hash-verified private SOURCE checkpoints."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.world_model.m3w_causal_descriptor_head import read_checkpoint

HOME = ROOT/"outputs/publication_readiness_2026_09/european_selection_controller_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    frozen = json.loads((HOME/"training_freeze.json").read_text()); rows = []
    if len(frozen["fits"]) != 18: raise ValueError("Complete original18fits required")
    for ref in frozen["fits"]:
        path = ROOT/ref["path"]
        if sha(path) != ref["sha256"]: raise ValueError("Training receipt changed")
        fit = json.loads(path.read_text()); cp = ROOT/fit["checkpoint"]["path"]
        if sha(cp) != fit["checkpoint"]["sha256"]: raise ValueError("SOURCE checkpoint changed")
        state = read_checkpoint(cp)
        if state["step"] != state["budget"] or state["unknown_supervised_rows"]:
            raise ValueError("Full known-only SOURCE cost training required")
        rows.append(dict(trial=fit["trial"],arm=fit["arm"],seed=fit["seed"],parameters=fit["parameters"],
            checkpoint_sha256=fit["checkpoint"]["sha256"],trace=state["trace"],
            source_sites=fit["source_sites"],source_only=True))
    result = dict(result_source="cached_verified_export_of_fresh_SOURCE_training", models=rows,
                  training_freeze_sha256=sha(HOME/"training_freeze.json"),
                  labels_or_weights_exported=False, selection_labels_read=False, new_training=False)
    path = HOME/"training_traces.json"
    if path.exists() and json.loads(path.read_text()) != result: raise ValueError("Scalar trace export immutable")
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(models=len(rows),scalar_trace_points=sum(len(r["trace"]) for r in rows),bytes=path.stat().st_size)))


if __name__ == "__main__": export()
