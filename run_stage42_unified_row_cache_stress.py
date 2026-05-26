from src.stage42_unified_row_cache_stress import build_unified_row_cache_stress


if __name__ == "__main__":
    result = build_unified_row_cache_stress()
    gate = result["stage42_ae_gate"]
    print(f"Stage42-AE unified row-cache stress: {gate['passed']} / {gate['total']} {gate['verdict']}")
