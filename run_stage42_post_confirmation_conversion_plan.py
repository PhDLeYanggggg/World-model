from src.stage42_post_confirmation_conversion_plan import run_stage42_post_confirmation_conversion_plan


if __name__ == "__main__":
    payload = run_stage42_post_confirmation_conversion_plan()
    gate = payload["stage42_gf_gate"]
    print(f"Stage42-GF post-confirmation conversion plan: {gate['verdict']} ({gate['passed']}/{gate['total']})")
