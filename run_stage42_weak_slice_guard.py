from src.stage42_weak_slice_guard import build_weak_slice_guard


if __name__ == "__main__":
    result = build_weak_slice_guard()
    gate = result["stage42_af_gate"]
    print(f"Stage42-AF weak-slice guard: {gate['passed']} / {gate['total']} {gate['verdict']}")
