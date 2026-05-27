from src.stage42_hz_to_cg_intake_bridge import run_stage42_hz_to_cg_intake_bridge


if __name__ == "__main__":
    result = run_stage42_hz_to_cg_intake_bridge()
    gate = result["stage42_ia_gate"]
    print(f"Stage42-IA HZ to CG intake bridge: {gate['verdict']} ({gate['passed']}/{gate['total']})")
