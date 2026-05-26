from src.stage42_repaired_protocol_robustness import run_stage42_repaired_protocol_robustness


if __name__ == "__main__":
    result = run_stage42_repaired_protocol_robustness()
    gate = result["stage42_ax_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
