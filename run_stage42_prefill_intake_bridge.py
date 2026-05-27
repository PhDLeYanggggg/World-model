from src.stage42_prefill_intake_bridge import run_stage42_prefill_intake_bridge


if __name__ == "__main__":
    payload = run_stage42_prefill_intake_bridge()
    gate = payload["stage42_gc_gate"]
    print(f"Stage42-GC prefill intake bridge: {gate['verdict']} ({gate['passed']}/{gate['total']})")
