from src.stage42_local_t100_schema_conversion import run_stage42_local_t100_schema_conversion


if __name__ == "__main__":
    result = run_stage42_local_t100_schema_conversion()
    gate = result["stage42_bf_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
