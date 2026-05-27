from src.stage42_conversion_capability_intake_bridge import run_stage42_conversion_capability_intake_bridge


if __name__ == "__main__":
    payload = run_stage42_conversion_capability_intake_bridge()
    gate = payload["stage42_ge_gate"]
    print(f"Stage42-GE conversion capability intake bridge: {gate['verdict']} ({gate['passed']}/{gate['total']})")
