from src.stage42_local_t100_conversion_readiness import run_stage42_local_t100_conversion_readiness


if __name__ == "__main__":
    result = run_stage42_local_t100_conversion_readiness()
    gate = result["stage42_be_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
