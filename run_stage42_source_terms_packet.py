from src.stage42_source_terms_packet import run_stage42_source_terms_packet


if __name__ == "__main__":
    result = run_stage42_source_terms_packet()
    gate = result["stage42_hz_gate"]
    print(f"Stage42-HZ source terms packet: {gate['verdict']} ({gate['passed']}/{gate['total']})")
