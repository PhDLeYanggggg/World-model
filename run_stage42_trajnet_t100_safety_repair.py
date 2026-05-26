from src.stage42_trajnet_t100_safety_repair import build_trajnet_t100_safety_repair


if __name__ == "__main__":
    result = build_trajnet_t100_safety_repair()
    gate = result["stage42_ai_gate"]
    print(f"Stage42-AI TrajNet t100 safety repair: {gate['passed']} / {gate['total']} {gate['verdict']}")
