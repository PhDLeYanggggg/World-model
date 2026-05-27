from src.stage42_ucy_h100_candidate_integrity_manifest import run_stage42_ucy_h100_candidate_integrity_manifest


if __name__ == "__main__":
    result = run_stage42_ucy_h100_candidate_integrity_manifest()
    gate = result["stage42_gx_gate"]
    print(f"Stage42-GX gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
