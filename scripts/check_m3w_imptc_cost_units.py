"""Hold geometry and forecasts fixed; perturb only two native-unit cost inputs."""
import json
from pathlib import Path
import platform
import sys

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 runtime required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.training.m3w_easy_moment import predict
from scripts.audit_m3w_imptc_input_contract import native_features
from scripts.run_m3w_native_forecast import immutable_json


def main():
    report = ROOT/"outputs/publication_readiness_2026_09/imptc_input_contract_v1"
    analysis = json.loads((report/"analysis.json").read_text())
    lineage = json.loads((report/"producer_chain.json").read_text())
    if file_digest(report/"producer_chain.json") != analysis["producer_chain_sha256"]:
        raise ValueError("Changed lineage receipt")
    artifact = analysis["private_arrays"]
    if file_digest(ROOT/artifact["path"]) != artifact["sha256"]:
        raise ValueError("Changed numerical probe inputs")
    results = {}
    with np.load(ROOT/artifact["path"], allow_pickle=False) as z:
        g, s = z["legacy_1p0_geometry"], z["legacy_1p0_scale"]
        for action in ("damped_velocity_005", "transformer", "eqmotion"):
            r = next(r for r in lineage["receipts"] if r["family"] == "moments" and f"coupa_seed17/{action}/" in r["receipt"])
            if file_digest(ROOT/r["checkpoint"]) != r["checkpoint_sha256"]:
                raise ValueError("Changed frozen head")
            cp = joblib.load(ROOT/r["checkpoint"])
            p = z["legacy_1p0_"+action+"_prediction"]
            x = native_features(g, p, s)
            np.testing.assert_array_equal(x, z["legacy_1p0_"+action+"_cost_features"])
            score = predict(cp["model"], x, cp["preprocess"])
            np.testing.assert_array_equal(score, z["legacy_1p0_"+action+"_moments"])
            results[action] = {}
            for factor in (.01, 100.):
                changed = native_features(g, p, s*factor)
                np.testing.assert_array_equal(changed[:, :354], x[:, :354])
                np.testing.assert_allclose(changed[:, 354]-x[:, 354], np.log(factor), rtol=1e-6, atol=2e-6)
                other = predict(cp["model"], changed, cp["preprocess"])
                results[action][str(factor)] = dict(
                    net_gain_sign_changed=int(((score[:, 0] > score[:, 1]) != (other[:, 0] > other[:, 1])).sum()),
                    maximum_moment_change=float(np.max(np.abs(other-score))),
                    fixed_normalized_feature_columns=354, changed_native_unit_columns=[354, 355])
    immutable_json(report/"cost_unit_mechanism_check.json", dict(
        result_source="fresh_run", code_sha256=file_digest(Path(__file__)),
        input_analysis_sha256=file_digest(report/"analysis.json"), rows=len(g), results=results,
        future_error_readout=False, tuning=False, complete_policy_evaluated=False,
        caveat="Net-gain sign is one risk-head signal, not scene-level deployment or realized gain"))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
